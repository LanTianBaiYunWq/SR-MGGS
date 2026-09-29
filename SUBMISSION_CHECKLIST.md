# Submission checklist

- [ ] Replace the anonymous paper title if the journal requires the final title.
- [ ] Add the final paper citation after acceptance.
- [ ] Add a journal-compatible software license.
- [ ] Remove author names, usernames, local paths, and institutional metadata
      for double-blind review.
- [x] Include the three-seed pretrained checkpoints in the release archive.
- [x] Add SHA-256 hashes and a release version.
- [ ] Confirm that the mixed OpenStreetMap/restricted-data statement is
      accepted by the target journal.
- [ ] Confirm that package identifiers associated with independently collected
      data do not reveal protected location or ownership information.
- [ ] Confirm that the data-use agreement permits public distribution of model
      weights trained using the independently collected data.
- [ ] Verify `conda env create -f environment.yml` on a clean machine.
- [ ] Run `python -m unittest tests/test_minimal_demo.py`.
- [ ] Run the minimal demo and compare it with `expected_output.json`.
- [ ] Re-run one strict independent calibration seed from the public package.
- [ ] Confirm that paper tables match `paper_results/`.
- [ ] Confirm that the manuscript uses SR-MGGS and multi-granularity terminology.
- [ ] Ensure that no unavailable experiment is described as completed.
- [ ] Create a versioned ZIP archive and record its SHA-256 checksum.

