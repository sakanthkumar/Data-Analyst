// Framer Motion centralized animation tokens (Executive & Calm style)

export const TRANSITIONS = {
  calm: {
    type: "tween",
    ease: [0.16, 1, 0.3, 1], // easeOutExponential
    duration: 0.3
  },
  springCalm: {
    type: "spring",
    stiffness: 140,
    damping: 20,
    mass: 1
  },
  springHover: {
    type: "spring",
    stiffness: 300,
    damping: 25
  }
};

export const VARIANTS = {
  fadeIn: {
    initial: { opacity: 0, y: 10 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -10 }
  },
  zoomIn: {
    initial: { scale: 0.96, opacity: 0 },
    animate: { scale: 1, opacity: 1 },
    exit: { scale: 0.96, opacity: 0 }
  },
  slideUp: {
    initial: { y: 20, opacity: 0 },
    animate: { y: 0, opacity: 1 },
    exit: { y: -20, opacity: 0 }
  },
  slideLeft: {
    initial: { x: 30, opacity: 0 },
    animate: { x: 0, opacity: 1 },
    exit: { x: 30, opacity: 0 }
  }
};
