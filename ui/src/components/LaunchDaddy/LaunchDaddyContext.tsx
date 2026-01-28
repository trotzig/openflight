import { useState, useCallback, useRef } from 'react';
import type { ReactNode } from 'react';
import { LaunchDaddyContext } from './launchDaddyTypes';

export function LaunchDaddyProvider({ children }: { children: ReactNode }) {
  const [isLaunchDaddyMode, setIsLaunchDaddyMode] = useState(false);
  const [isExploding, setIsExploding] = useState(false);
  const [secretTapCount, setSecretTapCount] = useState(0);
  const lastTapTime = useRef(0);

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
    if (now - lastTapTime.current > 2000) {
      setSecretTapCount(1);
    } else {
      setSecretTapCount(prev => {
        const newCount = prev + 1;
        if (newCount >= 5) {
          setIsLaunchDaddyMode(mode => !mode);
          return 0;
        }
        return newCount;
      });
    }
    lastTapTime.current = now;
  }, []);

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
