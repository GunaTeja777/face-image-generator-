# 🎭 Face Image Generator with Wasserstein GAN (WGAN)

This project is a deep learning web application that generates realistic human face images using a **Wasserstein GAN (WGAN)** trained on the **CelebA dataset**.

Users can click a button on the website to generate new AI-generated faces instantly.

---

## 🚀 Features

- Trained using **Wasserstein GAN** for improved stability
- Built with **PyTorch** for training and image generation
- Deployed using **Flask** backend and basic HTML/CSS frontend
- Hosted on **Render** for public access

---

## 🧠 Model Details

- **Model Type:** Wasserstein GAN (WGAN)
- **Generator Input:** 100-dimensional latent vector (random noise)
- **Output:** 64x64 RGB face image
- **Training Dataset:** [CelebA - CelebFaces Attributes Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- **Loss Function:** Wasserstein loss (no sigmoid in output layer)
- **Optimizer:** RMSprop / Adam (used with careful tuning)

---

## 📁 Project Structure

