
# 🎧 Audio Deepfake Detection Metrics

This repository provides a complete evaluation toolkit for **Audio Deepfake Detection** and **Speaker Verification** systems. It implements key evaluation metrics such as:

- ✅ Equal Error Rate (EER)
- ✅ Detection Cost Function (DCF)
- ✅ Tandem Detection Cost Function (t-DCF)  
  *(including constrained and unconstrained methods)*

The code is inspired by ASVspoof challenges and relevant academic literature in spoofing countermeasures and speaker verification evaluation.

---

## 📦 Dependencies

Make sure you have the following Python packages installed:

```bash
pip install numpy pandas scipy scikit-learn matplotlib torch
```

---

## ▶️ How to Run

```bash
python evaluation_measures.py
```

This will:
- Generate synthetic ASV and CM scores
- Compute evaluation metrics
- Print them to the console
- Plot t-DCF curves for visual analysis

---

## 🧠 Evaluation Theory

This repo implements metric computations based on the following concepts:

- **EER**: The point where false acceptance and false rejection rates are equal.
- **DCF**: Cost-based metric combining false accept/reject rates with task-specific costs.
- **t-DCF**: Measures the impact of a spoofing countermeasure (CM) in a tandem system, incorporating priors and detection costs.

### 🔗 Reference Paper

> T. Kinnunen, et al.,  
> *"t-DCF: a Detection Cost Function for the Tandem Assessment of Spoofing Countermeasures and Automatic Speaker Verification"*,  
> 	Published in Odyssey 2018: the Speaker and Language Recognition Workshop. [[PDF](https://www.isca-speech.org/archive/Odyssey_2018/pdfs/68.pdf)]
---
> T. Kinnunen, et al.,  
> *"Tandem Assessment of Spoofing Countermeasures and Automatic Speaker Verification: Fundamentals"*,  
> 	Published in IEEE/ACM Transactions on Audio, Speech, and Language Processing. [[PDF](https://doi.org/10.1109/TASLP.2020.3009494)]
---
To use the general a-DCF measure:
> T. Kinnunen, and Itshak Lapidot et al.,  
> *"Tandem Assessment of Spoofing Countermeasures and Automatic Speaker Verification: Fundamentals"*,  
> 	submitted to Speaker Odyssey 2024. [[PDF](https://doi.org/10.1109/TASLP.2020.3009494)], [[Github](https://github.com/shimhz/a_DCF)]

---

## 🙌 Acknowledgements

This work is inspired by the ASVspoof Challenge evaluation framework and research in spoofing countermeasures (CM) for automatic speaker verification (ASV) systems.

---
## 📚 Cite This Work

If you use this codebase in your research or publications, please consider citing it:

```bibtex
@misc{audio-deepfake-metrics,
  author       = {Avishai Weizman},
  title        = {Evaluation Measures for Audio Deepfake Detection and Speaker Verification},
  year         = {2025},
  url          = {https://github.com/avishai111/audio-deepfake-detection-metrics},
  note         = {GitHub repository}
}
```
---

## 📬 Contact

If you have questions, feedback, or want to collaborate, feel free to reach out:
*Avishai Weizman**  
 📧 Email: [Avishai11900@gmail.com](mailto:Avishai11900@gmail.com)  
 🔗 GitHub: [github.com/avishai111](https://github.com/avishai111)
 📄 arXiv: [arxiv.org/a/your_arxiv_id](https://arxiv.org/a/your_arxiv_id)  
 🎓 Google Scholar: [Avishai Weizman](https://scholar.google.com/citations?hl=iw&user=vWlnVpUAAAAJ)  
 💼 LinkedIn: [linkedin.com/in/avishai-weizman/](https://www.linkedin.com/in/avishai-weizman/)
 

