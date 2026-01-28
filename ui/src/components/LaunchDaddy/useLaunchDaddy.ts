import { useContext } from 'react';
import { LaunchDaddyContext } from './launchDaddyTypes';

export function useLaunchDaddy() {
  const context = useContext(LaunchDaddyContext);
  if (!context) {
    throw new Error('useLaunchDaddy must be used within LaunchDaddyProvider');
  }
  return context;
}
