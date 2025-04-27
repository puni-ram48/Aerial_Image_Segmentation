# Enhancing Aerial Image Segmentation: A Comparative Analysis of Machine Learning Approaches

This repository provides two segmentation pipelines for processing aerial images:

- **SegFormer-based Pretrained Segmentation**
- **SAM + Heuristic-based Classification**

Both pipelines are modular, inference-ready, and evaluated on standard segmentation metrics.

---

## 📍 SegFormer Pipeline

- Uses pre-trained **SegFormer-B5** model fine-tuned on Cityscapes.
- Processes input images, generates segmented outputs.
- Supports evaluation: Pixel Accuracy, IoU, Dice, Precision, Recall, and F1.

Scripts:
- `segformer_segmentation/pipeline.py` — Run segmentation
- `segformer_segmentation/visualization.py` — Visualize predictions vs ground truth
- `segformer_segmentation/evaluation.py` — Evaluate segmentation performance

Configuration:
- Modify `segformer_segmentation/config.json` to set paths and batch size.

---

## 📍 SAM + Heuristic Pipeline

- Uses **Segment Anything Model (SAM)** to generate masks.
- Applies **HSV color-based** and **contour-based** heuristics to classify regions into:
  - Road
  - Green Area
  - Building
- Evaluates segmentation with standard metrics.

Scripts:
- `sam_heuristic_segmentation/download_model.py` — Download the SAM model
- `sam_heuristic_segmentation/pipeline.py` — Run segmentation and classification
- `sam_heuristic_segmentation/visualization.py` — Visualize results
- `sam_heuristic_segmentation/evaluation.py` — Evaluate segmentation performance

Configuration:
- Modify `sam_heuristic_segmentation/config.py` to adjust settings.

---

## ⚙️ Installation

Clone the repository and install required packages:

```bash
git clone https://github.com/your-username/segmentation_project.git
cd segmentation_project
pip install -r requirements.txt
```

---

## 🚀 Running the Pipelines

### SegFormer Segmentation

```bash
cd segformer_segmentation
python pipeline.py
python visualization.py
python evaluation.py
```

---

### SAM + Heuristic Classification

```bash
cd sam_heuristic_segmentation
python download_model.py
python pipeline.py
python visualization.py
python evaluation.py
```

---

## 📊 Evaluation Metrics

Both pipelines evaluate segmentation performance using:

- Pixel Accuracy
- Intersection over Union (IoU)
- Dice Score
- Precision
- Recall
- F1 Score

Metrics are reported per-image and per-class, with overall summaries.

---

## 📢 Notes

- Input images are resized to **1024x1024** before processing.
- No model training is performed; pipelines are inference-only.
- Configurations (input paths, output paths, batch size, etc.) can be easily adjusted via config files.
- Ensure that filenames for input images and ground truth masks match exactly.

---

## 🤝 Contributing

We welcome contributions to improve the project!

To contribute:

1. Fork this repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes and commit them (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request describing your changes.

**Guidelines:**
- Keep code modular and clean.
- Document any new functionality clearly.
- Ensure compatibility with the existing project structure.
- Write meaningful commit messages.

Thank you for your contributions!

---

## 👨‍💻 Author

Developed by **Puneetha Dharmapura Shrirama**.

---

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---
