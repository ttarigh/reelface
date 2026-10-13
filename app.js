/**
 * THE MOST VIRAL FACE — FULLSCREEN 3D MASK & VERTICAL SCROLLING WALL
 */

(function () {
  'use strict';

  // State Management
  const state = {
    isDrifting: true,
    driftSpeed: 1.0,
    baseScrollY: 0,
    isDraggingBg: false,
    dragStartY: 0,
    dragOffsetY: 0,
    creators: [],
    // 3D Mask rotation angles
    targetRotX: 0,
    targetRotY: 0,
    currentRotX: 0,
    currentRotY: 0,
    isDraggingMask: false,
    maskDragStart: { x: 0, y: 0 },
  };

  // DOM Elements
  const elGridViewport = document.getElementById('grid-viewport');
  const elGridBg = document.getElementById('grid-tiled-bg');
  const elHighlightBox = document.getElementById('face-highlight-box');
  const elMaskWrapper = document.getElementById('mask-wrapper');
  const elTooltip = document.getElementById('creator-tooltip');
  const elTooltipRank = document.getElementById('tooltip-rank');
  const elTooltipHandle = document.getElementById('tooltip-handle');

  // Grid Constants: 25 columns, 40 rows = 1,000 faces
  let tileW = 64;
  let tileH = 80;

  function updateTileDimensions() {
    // If wide screen using 1600x3200 background size
    if (window.innerWidth >= 1200) {
      tileW = 1600 / 25; // 64px
      tileH = 3200 / 40; // 80px
    } else {
      tileW = window.innerWidth / 25;
      tileH = tileW * 1.25;
    }
  }
  updateTileDimensions();
  window.addEventListener('resize', updateTileDimensions);

  // ==========================================================================
  // 1. LOAD CREATOR REEL MANIFEST
  // ==========================================================================
  async function loadCreators() {
    try {
      const res = await fetch('assets/creators.json');
      state.creators = await res.json();
    } catch (err) {
      console.warn('Could not load creators.json', err);
    }
  }

  // ==========================================================================
  // 2. SMOOTH VERTICAL-ONLY DRIFT (UP AND DOWN, NO DIAGONAL)
  // ==========================================================================
  let lastTime = performance.now();

  function animate(now) {
    const dt = (now - lastTime) / 1000;
    lastTime = now;

    if (state.isDrifting && !state.isDraggingBg) {
      // Continuous pure vertical drift (strictly along the Y-axis)
      const verticalSpeed = 48 * state.driftSpeed; // 48px per second
      state.baseScrollY += verticalSpeed * dt;
      if (state.baseScrollY >= 3200) {
        state.baseScrollY -= 3200;
      }
    }

    const currentY = state.baseScrollY + state.dragOffsetY;
    // Set position strictly on Y axis, center-aligned on X axis
    if (window.innerWidth >= 1200) {
      elGridBg.style.backgroundPosition = `center ${currentY}px`;
    } else {
      elGridBg.style.backgroundPosition = `0px ${currentY}px`;
    }

    // Smooth 3D Mask rotation with subtle spring lerp
    const lerp = 0.08;
    state.currentRotX += (state.targetRotX - state.currentRotX) * lerp;
    state.currentRotY += (state.targetRotY - state.currentRotY) * lerp;

    const rotX = state.currentRotX.toFixed(2);
    const rotY = state.currentRotY.toFixed(2);
    elMaskWrapper.style.transform = `rotateX(${rotX}deg) rotateY(${rotY}deg) translateZ(40px)`;

    requestAnimationFrame(animate);
  }

  // Vertical Drag & Pan on Background
  elGridViewport.addEventListener('mousedown', (e) => {
    if (e.target.closest('#mask-wrapper')) return;
    state.isDraggingBg = true;
    state.dragStartY = e.clientY;
    elGridViewport.classList.add('grabbing');
  });

  window.addEventListener('mousemove', (e) => {
    if (state.isDraggingBg) {
      const dy = e.clientY - state.dragStartY;
      state.baseScrollY += dy;
      state.dragStartY = e.clientY;
      return;
    }

    // 3D Perspective Tilt tracking from cursor position
    if (!state.isDraggingMask) {
      const w = window.innerWidth;
      const h = window.innerHeight;
      const normX = (e.clientX - w / 2) / (w / 2); // -1 to +1
      const normY = (e.clientY - h / 2) / (h / 2); // -1 to +1

      state.targetRotY = normX * 24;
      state.targetRotX = -normY * 20;
    }

    // Identify creator face under cursor
    handleBackgroundFaceHover(e);
  });

  window.addEventListener('mouseup', () => {
    if (state.isDraggingBg) {
      state.isDraggingBg = false;
      elGridViewport.classList.remove('grabbing');
    }
    if (state.isDraggingMask) {
      state.isDraggingMask = false;
    }
  });

  // Direct 3D Orbital Dragging on the Mask itself
  elMaskWrapper.addEventListener('mousedown', (e) => {
    e.stopPropagation();
    state.isDraggingMask = true;
    state.maskDragStart = { x: e.clientX, y: e.clientY };
  });

  window.addEventListener('mousemove', (e) => {
    if (state.isDraggingMask) {
      const dx = e.clientX - state.maskDragStart.x;
      const dy = e.clientY - state.maskDragStart.y;
      state.targetRotY += dx * 0.45;
      state.targetRotX -= dy * 0.45;
      state.maskDragStart = { x: e.clientX, y: e.clientY };
    }
  });

  // Device Orientation (Gyroscope tilt on mobile/tablets)
  if (window.DeviceOrientationEvent) {
    window.addEventListener('deviceorientation', (e) => {
      if (e.gamma !== null && e.beta !== null && !state.isDraggingMask) {
        state.targetRotY = Math.max(-25, Math.min(25, e.gamma * 0.7));
        state.targetRotX = Math.max(-22, Math.min(22, (e.beta - 45) * 0.6));
      }
    }, { passive: true });
  }

  // ==========================================================================
  // 3. HOVER IDENTIFICATION & CLICKING DIRECTLY INTO THE REEL
  // ==========================================================================
  function getCreatorFromEvent(e) {
    if (!state.creators || state.creators.length === 0) return null;

    const bgRect = elGridBg.getBoundingClientRect();
    const relX = (e.clientX - bgRect.left);
    let relY = (e.clientY - bgRect.top);

    if (relX < 0 || relX >= bgRect.width) return null;

    const col = Math.floor(relX / tileW) % 25;
    const row = Math.floor(relY / tileH) % 40;
    const index = (row * 25 + col) % 1000;

    return { creator: state.creators[index], col, row, relX, relY, bgRect };
  }

  function handleBackgroundFaceHover(e) {
    if (e.target.closest('#mask-wrapper')) {
      elTooltip.style.opacity = '0';
      elHighlightBox.style.opacity = '0';
      return;
    }

    const data = getCreatorFromEvent(e);
    if (data && data.creator) {
      const { creator, col, row, bgRect } = data;
      elTooltipRank.textContent = `#${creator.rank}`;
      elTooltipHandle.textContent = `@${creator.handle}`;
      elTooltip.style.left = `${e.clientX}px`;
      elTooltip.style.top = `${e.clientY}px`;
      elTooltip.style.opacity = '1';

      const boxLeft = bgRect.left + col * tileW;
      const boxTop = bgRect.top + (Math.floor((e.clientY - bgRect.top) / tileH) * tileH);
      elHighlightBox.style.width = `${tileW}px`;
      elHighlightBox.style.height = `${tileH}px`;
      elHighlightBox.style.transform = `translate3d(${boxLeft}px, ${boxTop}px, 0)`;
      elHighlightBox.style.opacity = '0.9';
    } else {
      elTooltip.style.opacity = '0';
      elHighlightBox.style.opacity = '0';
    }
  }

  // Clicking on any creator face opens the Reel directly (not the creator's page)
  elGridViewport.addEventListener('click', (e) => {
    if (e.target.closest('#mask-wrapper')) return;

    const data = getCreatorFromEvent(e);
    if (data && data.creator) {
      const reelUrl = data.creator.reelUrl;
      if (reelUrl) {
        window.open(reelUrl, '_blank');
      }
    }
  });

  // Spacebar to pause/resume vertical drift
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
      e.preventDefault();
      state.isDrifting = !state.isDrifting;
    } else if (e.code === 'Escape') {
      state.targetRotX = 0;
      state.targetRotY = 0;
    }
  });

  // Initialize
  loadCreators();
  requestAnimationFrame(animate);

})();
