# Application form questions — VERBATIM (captured 2026-08-31)

> Neel reads these answers for EVERY applicant and filters on them before
> deciding whether the write-up gets opened. These are the highest-leverage
> words in the submission. Source: the live application form, copied by the
> owner. Do not paraphrase the questions; draft against them exactly.

**Deliverable format implied by the form:** ONE Google Doc, link-viewable by
anyone, containing the executive summary as the FIRST 1–3 PAGES followed by
the main write-up. Applications without a doc are rejected. Optional: links
to other outputs (code, colab, etc) — our GitHub repo goes here.

## The questions

1. Link a Google Doc with your executive summary and main project write-up
   (anyone-with-link viewable; first 1–3 pages = executive summary).
   (Optional) Link to any other relevant outputs (code, colab, etc).

2. **What question did you try to answer?**

3. **Why is this question interesting / why did you choose it?**

4. **What conclusions have you reached about this research problem?**

5. **Technical setup: What are the key things you try to quantify in this
   study and how do you define and measure them? Give the key technical
   details: what models you use, datasets, prompts, the metrics used.**
   (Example things you might try to quantify: deception, faithfulness of
   CoT, model confidence, model confusion)

6. **What is the strongest evidence you found AGAINST these hypotheses?**

7. **What are the biggest limitations to your results? Could you have
   addressed them?**
   ("Please be honest! It's much better to flag a limitation yourself than
   for me to need to figure it out.")

8. **How did you use LLMs in this research task and write-up? Which LLMs?
   How exactly did you make sure that they weren't just giving you slop?**
   ("Please explain in detail, including which parts you did and didn't
   check, how you prioritized, and how surprised you'd be to discover a
   major error in each part.")

9. **What, if any, prior experience do you have with mechanistic
   interpretability?**

10. **Other than your research task, what are 1–3 pieces of evidence that
    you'd be able to do good research in the program?** (~100 words; not
    standard credentials; must NOT just point at the project.)

11. **Why are you interested in Neel's stream specifically?**

12. **What is the likelihood you will join the training program
    (Sept 28 – Oct 30) if accepted?**

13. (Optional) **Anything else important about your application project not
    covered above?**

## Mapping to our record (drafting notes)

- Q2/Q4/Q5 → EXPERIMENT.md + RESULTS.md; Q4 must carry the corrected,
  calibrated conclusions (Runs 007–009), not the withdrawn headline.
- Q6 is a GIFT for this project: we refuted our own headline (Run 007), our
  own sealed forecast (pre-reg I), and our own doubt of the second forecast
  (Run 008). Few applicants will have real content here.
- Q7 → RESULTS.md limitations: n_test/negatives, one model one dataset,
  population confound in Run 008, judge memorization, correctness≠alignment.
  "Could you have addressed them?" → yes: more negatives was affordable;
  say so plainly.
- Q8 is aimed exactly at how this project was run: agent-built pipeline +
  independent verification + mutation-tested suite + adversarial red-team
  that killed the headline + the hand-verification checklists. Answer with
  specifics, including what was NOT checked and where a major error would
  still surprise us least (the write-up must name one honestly).
- Q9, Q10, Q11, Q12 → OWNER ONLY. Tyler must not fabricate these.
- Q13 → candidates: the pre-registration commits (sealed before data, in
  git with timestamps), the public repo, the red-team scripts.
