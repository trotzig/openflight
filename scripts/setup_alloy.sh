#!/bin/bash
#
# Setup Grafana Alloy on Raspberry Pi for OpenFlight log shipping
#
# This script installs Grafana Alloy and configures it to tail OpenFlight
# session logs and ship them to Grafana Cloud Loki.
#
# Usage:
#   ./setup_alloy.sh
#
# Prerequisites:
#   - Raspberry Pi running Raspberry Pi OS (Debian-based)
#   - Grafana Cloud account with Loki endpoint
#
# After running this script, create /etc/alloy/credentials.env with:
#   LOKI_URL=https://logs-prod-xxx.grafana.net/loki/api/v1/push
#   LOKI_USER=your-instance-id
#   LOKI_API_KEY=your-api-key

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== OpenFlight Alloy Setup ===${NC}"

# Detect architecture
ARCH=$(dpkg --print-architecture)
echo "Detected architecture: $ARCH"

# Get the current user's home directory for log path
OPENFLIGHT_USER=${SUDO_USER:-$USER}
OPENFLIGHT_HOME=$(eval echo ~$OPENFLIGHT_USER)
LOG_DIR="$OPENFLIGHT_HOME/openflight_sessions"

echo "OpenFlight user: $OPENFLIGHT_USER"
echo "Log directory: $LOG_DIR"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run as root (sudo)${NC}"
    exit 1
fi

# Step 1: Add Grafana APT repository
echo -e "\n${GREEN}[1/5] Adding Grafana APT repository...${NC}"
apt-get install -y apt-transport-https software-properties-common wget

mkdir -p /etc/apt/keyrings/
wget -q -O - https://apt.grafana.com/gpg.key | gpg --dearmor > /etc/apt/keyrings/grafana.gpg

echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" | tee /etc/apt/sources.list.d/grafana.list

apt-get update

# Step 2: Install Alloy
echo -e "\n${GREEN}[2/5] Installing Grafana Alloy...${NC}"
apt-get install -y alloy

# Step 3: Create Alloy configuration
echo -e "\n${GREEN}[3/5] Creating Alloy configuration...${NC}"

# Create config directory if needed
mkdir -p /etc/alloy

# Write main configuration
cat > /etc/alloy/config.alloy << EOF
// OpenFlight Log Shipping Configuration
// Ships session logs to Grafana Cloud Loki

// Discover JSONL session log files
local.file_match "openflight_logs" {
  path_targets = [{"__path__" = "${LOG_DIR}/session_*.jsonl"}]
  sync_period  = "10s"
}

// Tail log files
loki.source.file "openflight" {
  targets    = local.file_match.openflight_logs.targets
  forward_to = [loki.process.openflight.receiver]

  // Track position to survive restarts
  tail_from_end = false
}

// Process and enrich log entries
loki.process "openflight" {
  forward_to = [loki.write.grafana_cloud.receiver]

  // Parse JSON log entries
  stage.json {
    expressions = {
      log_type = "type",
      mode     = "mode",
      club     = "club",
      shot_num = "shot_number",
    }
  }

  // Add extracted values as labels
  stage.labels {
    values = {
      log_type = "",
      mode     = "",
      club     = "",
    }
  }

  // Add static labels
  stage.static_labels {
    values = {
      app  = "openflight",
      host = constants.hostname,
    }
  }

  // Extract timestamp from JSON ts field
  stage.json {
    expressions = {
      timestamp = "ts",
    }
  }

  stage.timestamp {
    source = "timestamp"
    format = "RFC3339"
  }
}

// Ship to Grafana Cloud Loki
loki.write "grafana_cloud" {
  endpoint {
    url = env("LOKI_URL")

    basic_auth {
      username = env("LOKI_USER")
      password = env("LOKI_API_KEY")
    }
  }

  // Buffer logs locally if network is unavailable
  external_labels = {}
}
EOF

echo "Configuration written to /etc/alloy/config.alloy"

# Step 4: Create credentials template
echo -e "\n${GREEN}[4/5] Creating credentials template...${NC}"

if [ ! -f /etc/alloy/credentials.env ]; then
    cat > /etc/alloy/credentials.env << 'EOF'
# Grafana Cloud Loki Credentials
# Get these from: Grafana Cloud -> Your Stack -> Loki -> Details
#
# LOKI_URL: The push endpoint URL
# LOKI_USER: Your Grafana Cloud instance ID (numeric)
# LOKI_API_KEY: API key with logs:write scope

LOKI_URL=https://logs-prod-us-central1.grafana.net/loki/api/v1/push
LOKI_USER=123456
LOKI_API_KEY=your-api-key-here
EOF
    chmod 600 /etc/alloy/credentials.env
    echo -e "${YELLOW}Created /etc/alloy/credentials.env - please edit with your Grafana Cloud credentials${NC}"
else
    echo "Credentials file already exists, not overwriting"
fi

# Step 5: Configure systemd service
echo -e "\n${GREEN}[5/5] Configuring systemd service...${NC}"

mkdir -p /etc/systemd/system/alloy.service.d

cat > /etc/systemd/system/alloy.service.d/override.conf << EOF
[Service]
# Load Grafana Cloud credentials
EnvironmentFile=/etc/alloy/credentials.env

# Run as the openflight user to access log files
User=$OPENFLIGHT_USER
Group=$OPENFLIGHT_USER

# Ensure we can read log files
ReadOnlyPaths=$LOG_DIR
EOF

# Reload systemd and enable service
systemctl daemon-reload
systemctl enable alloy

echo -e "\n${GREEN}=== Setup Complete ===${NC}"
echo ""
echo "Next steps:"
echo "1. Edit /etc/alloy/credentials.env with your Grafana Cloud credentials"
echo "   - Get these from: Grafana Cloud -> Your Stack -> Loki -> Details"
echo ""
echo "2. Start the Alloy service:"
echo "   sudo systemctl start alloy"
echo ""
echo "3. Check service status:"
echo "   sudo systemctl status alloy"
echo "   sudo journalctl -u alloy -f"
echo ""
echo "4. Verify logs in Grafana Cloud:"
echo "   Query: {app=\"openflight\"}"
echo ""
echo -e "${YELLOW}Note: Alloy will start shipping logs once credentials are configured${NC}"
