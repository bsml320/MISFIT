(function (root, factory) {
  const core = factory();
  if (typeof module === 'object' && module.exports) module.exports = core;
  else root.MisfitCore = core;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const FPS = 20;
  const DURATION = 45;
  const COHORTS = Object.freeze([
    Object.freeze({ id: 'oasis3', label: 'OASIS-3', start: 0, end: 15,
      sample: '1,360 participants', detail: 'Voxelwise associations' }),
    Object.freeze({ id: 'adni', label: 'ADNI', start: 15, end: 30,
      sample: '731 acquisitions · 547 participants', detail: 'Participant-purged fits' }),
    Object.freeze({ id: 'oasis1', label: 'OASIS-1', start: 30, end: 45,
      sample: '226 participants', detail: 'Schaefer-100 parcel associations' })
  ]);
  function clampTime(time) {
    return Math.min(DURATION - 1 / FPS, Math.max(0, Number.isFinite(time) ? time : 0));
  }
  function cohortAt(time) {
    const bounded = clampTime(time);
    return COHORTS.find(item => bounded >= item.start && bounded < item.end);
  }
  function stepTime(time, direction) {
    return clampTime(Math.round((clampTime(time) * FPS + Math.sign(direction))) / FPS);
  }
  function formatTime(time) {
    const seconds = Math.max(0, Math.floor(Number.isFinite(time) ? time : 0));
    return `${String(Math.floor(seconds / 60)).padStart(2,'0')}:${String(seconds % 60).padStart(2,'0')}`;
  }
  return Object.freeze({ FPS, DURATION, COHORTS, clampTime, cohortAt, stepTime, formatTime });
});
