const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const corePath = path.join(__dirname, '..', 'viewer-core.js');

test('the viewer navigation module exists', () => {
  assert.equal(fs.existsSync(corePath), true, 'viewer navigation module is missing');
});

if (fs.existsSync(corePath)) {
  const core = require(corePath);
  test('cohorts cover the original chapters and exact boundaries', () => {
    assert.deepEqual(core.COHORTS.map(x => [x.id, x.start, x.end]), [
      ['oasis3', 0, 15], ['adni', 15, 30], ['oasis1', 30, 45]
    ]);
    for (const [time, id] of [[0,'oasis3'],[14.999,'oasis3'],[15,'adni'],
      [29.999,'adni'],[30,'oasis1'],[45,'oasis1']]) {
      assert.equal(core.cohortAt(time).id, id);
    }
  });
  test('non-finite and out-of-range times are bounded', () => {
    assert.equal(core.clampTime(-5), 0);
    assert.equal(core.clampTime(100), 44.95);
    assert.equal(core.clampTime(NaN), 0);
    assert.equal(core.clampTime(Infinity), 0);
  });
  test('frame stepping follows 20 fps without exceeding the movie', () => {
    assert.equal(core.stepTime(7.5, 1), 7.55);
    assert.equal(core.stepTime(7.5, -1), 7.45);
    assert.equal(core.stepTime(0, -1), 0);
    assert.equal(core.stepTime(44.95, 1), 44.95);
    assert.equal(core.stepTime(14.95, 1), 15);
  });
  test('timestamps remain readable at the end and on invalid input', () => {
    assert.equal(core.formatTime(0), '00:00');
    assert.equal(core.formatTime(7.5), '00:07');
    assert.equal(core.formatTime(45), '00:45');
    assert.equal(core.formatTime(NaN), '00:00');
  });
}
