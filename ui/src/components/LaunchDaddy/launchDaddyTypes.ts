import { createContext } from 'react';

export interface LaunchDaddyContextType {
  isLaunchDaddyMode: boolean;
  toggleLaunchDaddy: () => void;
  triggerExplosion: () => void;
  isExploding: boolean;
  secretTapCount: number;
  handleSecretTap: () => void;
}

export const LaunchDaddyContext = createContext<LaunchDaddyContextType | null>(null);
