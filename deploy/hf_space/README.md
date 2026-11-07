---
title: VisionGuard API
emoji: 🛡️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: AI-generated vs real image classifier (model API)
---

# VisionGuard model API

JSON API for the VisionGuard classifier (MobileNetV2, registry model `visionguard` v2).

| Method | Path | Auth | Body |
|---|---|---|---|
| GET | `/health` | none | — |
| POST | `/predict` | header `X-API-Key` | multipart form field `image` (JPG/PNG, ≤ 5 MB) |

Source code: https://github.com/RizwanArshad845/ML_A2
