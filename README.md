# Enhancing Aerial Image Segmentation: A Comparative Analysis of Machine Learning Approaches

## Project Description

This project provides two complete pipelines for semantic segmentation of aerial imagery:

- **SegFormer-based Pretrained Segmentation**  
- **SAM + Heuristic-based Classification**

The purpose of this project is to efficiently perform segmentation without model training by leveraging powerful pretrained models. It enables users to generate semantic masks for road, green area, and building classes and evaluate their segmentation quality using robust metrics.

### Key Features

- 📌 Pretrained SegFormer-B5 model for fast and accurate inference.
- 📌 Segment Anything Model (SAM) integration with heuristic-based classification.
- 📌 Professional evaluation pipeline: Pixel Accuracy, IoU, Dice Score, Precision, Recall, and F1 Score.
- 📌 Modular code structure for easy customization.

---

## ⚙️ Installation Instructions

*  Clone the repository:

```bash
git clone https://github.com/puni-ram48/Aerial_Image_Segmentation.git
cd Aerial_Image_Segmentation
```
---

##  Usage Examples

### 1. SegFormer-based Segmentation

Navigate to the SegFormer pipeline folder:

```bash
cd segformer_segmentation
```
Install required dependencies:

```bash
pip install -r requirements.txt
```
- Update `config.json` as needed (input/output paths).
- Run the segmentation:

```bash
python pipeline.py
```

- Visualize the segmented outputs:

```bash
python visualization.py
```

- Evaluate the segmentation performance:

```bash
python evaluation.py
```

---

### 2. SAM + Heuristic-based Classification

Navigate to the SAM pipeline folder:

```bash
cd sam_heuristic_segmentation
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

- Download the SAM model weights:

```bash
python download_model.py
```

- Run the segmentation and heuristic classification:

```bash
python pipeline.py
```

- Visualize outputs:

```bash
python visualization.py
```

- Evaluate the results:

```bash
python evaluation.py
```
---
## Contributing
We welcome contributions to this project! If you would like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes and commit them (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Create a new Pull Request.

Please ensure your code is well-documented.

## Authors and Acknowledgment
This project was initiated and completed by Puneetha Dharmapura Shrirama. 

Special thanks to our supervisor **[Thomas Maier]** for their valuable guidance and support throughout the project.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.
