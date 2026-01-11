import { createContext, useContext, useState, useCallback } from 'react';
import type { ReactNode } from 'react';

interface LaunchDaddyContextType {
  isLaunchDaddyMode: boolean;
  toggleLaunchDaddy: () => void;
  triggerExplosion: () => void;
  isExploding: boolean;
  secretTapCount: number;
  handleSecretTap: () => void;
}

const LaunchDaddyContext = createContext<LaunchDaddyContextType | null>(null);

export function LaunchDaddyProvider({ children }: { children: ReactNode }) {
  const [isLaunchDaddyMode, setIsLaunchDaddyMode] = useState(false);
  const [isExploding, setIsExploding] = useState(false);
  const [secretTapCount, setSecretTapCount] = useState(0);
  const [lastTapTime, setLastTapTime] = useState(0);

  const toggleLaunchDaddy = useCallback(() => {
    setIsLaunchDaddyMode(prev => !prev);
    setSecretTapCount(0);
  }, []);

  const triggerExplosion = useCallback(() => {
    if (!isLaunchDaddyMode) return;
    setIsExploding(true);
    setTimeout(() => setIsExploding(false), 2500);
  }, [isLaunchDaddyMode]);

  // Secret activation: tap 5 times quickly on the logo
  const handleSecretTap = useCallback(() => {
    const now = Date.now();
    if (now - lastTapTime > 2000) {
      // Reset if more than 2 seconds since last tap
      setSecretTapCount(1);
    } else {
      setSecretTapCount(prev => {
        const newCount = prev + 1;
        if (newCount >= 5) {
          setIsLaunchDaddyMode(prev => !prev);
          return 0;
        }
        return newCount;
      });
    }
    setLastTapTime(now);
  }, [lastTapTime]);

  return (
    <LaunchDaddyContext.Provider
      value={{
        isLaunchDaddyMode,
        toggleLaunchDaddy,
        triggerExplosion,
        isExploding,
        secretTapCount,
        handleSecretTap,
      }}
    >
      {children}
    </LaunchDaddyContext.Provider>
  );
}

export function useLaunchDaddy() {
  const context = useContext(LaunchDaddyContext);
  if (!context) {
    throw new Error('useLaunchDaddy must be used within LaunchDaddyProvider');
  }
  return context;
}
