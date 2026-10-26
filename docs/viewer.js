(function () {
  'use strict';
  const core = window.MisfitCore;
  const video = document.getElementById('brain-video');
  if (!core || !video) return;
  const timeline = document.getElementById('timeline');
  const playButton = document.getElementById('play-pause');
  const errorBox = document.getElementById('error-message');
  const fullscreen = document.getElementById('fullscreen');
  const buttons = Array.from(document.querySelectorAll('[data-cohort]'));
  let pendingSeek = null;
  let lastCohort = '';

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
  }
  function update() {
    const time = video.currentTime || 0;
    timeline.value = String(core.clampTime(time));
    timeline.setAttribute('aria-valuetext', `${core.formatTime(time)} of 00:45`);
    document.getElementById('current-time').textContent = core.formatTime(time);
    playButton.textContent = video.paused ? 'Play' : 'Pause';
    playButton.setAttribute('aria-label', video.paused ? 'Play video' : 'Pause video');
    const cohort = core.cohortAt(time);
    if (cohort.id !== lastCohort) {
      lastCohort = cohort.id;
      document.getElementById('cohort-status').textContent = cohort.label;
      document.getElementById('cohort-sample').textContent = cohort.sample;
      for (const button of buttons) {
        const selected = button.dataset.cohort === cohort.id;
        button.classList.toggle('selected', selected);
        button.setAttribute('aria-pressed', String(selected));
      }
    }
  }
  function seek(time, pause) {
    if (pause) video.pause();
    const target = core.clampTime(time);
    if (video.readyState === 0) {
      pendingSeek = target;
      video.load();
    } else video.currentTime = target;
    update();
  }
  async function togglePlayback() {
    if (!video.paused) { video.pause(); return; }
    try {
      if (video.ended) seek(0, false);
      await video.play();
      errorBox.hidden = true;
    } catch (error) {
      showError('Playback could not start. Try Play again, or use the MP4 download link below.');
    }
    update();
  }
  playButton.addEventListener('click', togglePlayback);
  timeline.addEventListener('input', () => seek(Number(timeline.value), true));
  document.getElementById('previous-frame').addEventListener('click', () => seek(core.stepTime(video.currentTime,-1),true));
  document.getElementById('next-frame').addEventListener('click', () => seek(core.stepTime(video.currentTime,1),true));
  document.getElementById('restart').addEventListener('click', () => seek(core.cohortAt(video.currentTime).start,true));
  document.getElementById('speed').addEventListener('change', event => {
    video.playbackRate = Number(event.target.value);
  });
  for (const button of buttons) button.addEventListener('click', () => {
    const cohort = core.COHORTS.find(item => item.id === button.dataset.cohort);
    seek(cohort.start, true);
  });
  const stage = document.getElementById('video-stage');
  fullscreen.disabled = !(stage.requestFullscreen || video.webkitEnterFullscreen);
  fullscreen.addEventListener('click', async () => {
    try {
      if (stage.requestFullscreen) await stage.requestFullscreen();
      else if (video.webkitEnterFullscreen) video.webkitEnterFullscreen();
    } catch (error) { showError('Full-screen viewing is unavailable in this browser. You can still use the player or download the MP4.'); }
  });
  document.addEventListener('fullscreenchange', () => {
    video.controls = Boolean(document.fullscreenElement);
  });
  video.addEventListener('webkitbeginfullscreen', () => { video.controls = true; });
  video.addEventListener('webkitendfullscreen', () => { video.controls = false; });
  video.tabIndex = 0;
  video.addEventListener('keydown', event => {
    if (event.code === 'Space') { event.preventDefault(); togglePlayback(); }
  });
  for (const name of ['timeupdate','seeked','play','pause','ended','loadeddata']) video.addEventListener(name, update);
  video.addEventListener('loadedmetadata', () => {
    if (pendingSeek !== null) { video.currentTime = pendingSeek; pendingSeek = null; }
    update();
  });
  video.addEventListener('error', () => showError('The video could not be loaded. Keep the assets folder beside index.html, or try the MP4 download link below.'));
  document.getElementById('custom-controls').hidden = false;
  video.controls = false;
  update();
})();
