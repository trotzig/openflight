import { useMemo } from 'react';
import { useLaunchDaddy } from './LaunchDaddyContext';

// Generate random particles for the explosion
function generateParticles(count: number) {
  return Array.from({ length: count }, (_, i) => {
    const angle = (Math.random() * 360) * (Math.PI / 180);
    const distance = 100 + Math.random() * 400;
    const tx = Math.cos(angle) * distance;
    const ty = Math.sin(angle) * distance;
    const type = ['fire', 'spark', 'ember'][Math.floor(Math.random() * 3)];
    const delay = Math.random() * 0.3;
    const duration = 0.8 + Math.random() * 0.7;

    return {
      id: i,
      tx,
      ty,
      type,
      delay,
      duration,
      startX: 45 + Math.random() * 10, // percentage
      startY: 55 + Math.random() * 10, // percentage
    };
  });
}

// Epic phrases for the stratosphere text
const LAUNCH_PHRASES = [
  'STRATOSPHERE',
  'ORBIT ACHIEVED',
  'TO THE MOON',
  'NUCLEAR LAUNCH',
  'ABSOLUTE BOMB',
  'INTO ORBIT',
  'LAUNCHED',
  'OBLITERATED',
  'VAPORIZED',
];

const SUBTITLES = [
  'THE LONGEST HITTER IN THE WORLD',
  'THAT BALL HAD A FAMILY',
  'CALL THE AUTHORITIES',
  'WEAPONS GRADE DISTANCE',
  'SPONSORED BY NASA',
  'REGISTERED AS A WEAPON',
  'BROKE THE SOUND BARRIER',
];

export function LaunchDaddyOverlay() {
  const { isExploding, isLaunchDaddyMode } = useLaunchDaddy();

  const particles = useMemo(() => generateParticles(50), [isExploding]);
  const phrase = useMemo(
    () => LAUNCH_PHRASES[Math.floor(Math.random() * LAUNCH_PHRASES.length)],
    [isExploding]
  );
  const subtitle = useMemo(
    () => SUBTITLES[Math.floor(Math.random() * SUBTITLES.length)],
    [isExploding]
  );

  if (!isLaunchDaddyMode || !isExploding) return null;

  return (
    <>
      {/* Fire border effect */}
      <div className="launch-daddy-fire-border" />

      {/* Explosion flash and rings */}
      <div className="launch-daddy-explosion">
        <div className="launch-daddy-explosion__flash" />
        <div className="launch-daddy-explosion__ring" />
        <div className="launch-daddy-explosion__ring" />
        <div className="launch-daddy-explosion__ring" />

        {/* Particle system */}
        <div className="launch-daddy-explosion__particles">
          {particles.map((particle) => (
            <div
              key={particle.id}
              className={`launch-daddy-particle launch-daddy-particle--${particle.type}`}
              style={{
                left: `${particle.startX}%`,
                top: `${particle.startY}%`,
                '--tx': `${particle.tx}px`,
                '--ty': `${particle.ty}px`,
                animationDelay: `${particle.delay}s`,
                animationDuration: `${particle.duration}s`,
              } as React.CSSProperties}
            />
          ))}
        </div>
      </div>

      {/* Stratosphere launch */}
      <div className="launch-daddy-stratosphere">
        <div className="launch-daddy-stratosphere__ball" />
        <div className="launch-daddy-stratosphere__text">{phrase}</div>
        <div className="launch-daddy-stratosphere__subtitle">{subtitle}</div>
      </div>
    </>
  );
}
