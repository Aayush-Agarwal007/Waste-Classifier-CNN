# 🌍 WasteAI - Smart Waste Segregation

An intelligent waste classification system powered by Deep Learning that identifies and categorizes waste items, providing disposal guidance and financial incentives through real-time value calculation.

---

## ⚡ Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/Aayush-Agarwal007/waste_Prediction-_System_using_CNN.git
cd waste_Prediction-_System_using_CNN

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

**Opens at**: http://localhost:8501

---

## 🎯 Features

### 1. **Smart Classification**
- 🔍 Identifies waste into 6 categories
- 📊 Confidence scores (0-100%)
- ⚠️ Low-confidence alerts
- 📈 Probability distribution chart

### 2. **Disposal Guidance**
- 🗑️ Bin color recommendations
- ✅ Recyclability status
- ⏱️ Decomposition times
- 📋 Step-by-step disposal steps
- 🌱 Environmental facts

### 3. **Environmental Metrics**
- 🌍 CO₂ savings per item
- 💧 Water conservation data
- ⚡ Energy savings calculations

### 4. **💰 Cost Calculator** ⭐ NEW
- **Real-time Value**: Adjust weight and see instant calculations
- **Resale Value**: Market prices for each waste type
- **Carbon Credits**: Monetize environmental impact
- **Bulk Estimation**: Estimate value of sorting multiple items
- **Achievements**: Unlock badges for high-value contributions

**Example**: 100g metal waste = $0.08 total value (resale + carbon credits)

---

## 📚 Waste Categories

| Type | Bin | Recyclable | Time to Decompose | Market Value |
|------|-----|-----------|-------------------|--------------|
| Cardboard | 🔵 Blue | ✅ | 2-3 months | $0.05/kg |
| Glass | 🟢 Green | ✅ | 1M years | $0.03/kg |
| Metal | 🟡 Yellow | ✅ | 200-500 years | $0.50/kg ⭐ |
| Paper | 🔵 Blue | ✅ | 2-6 weeks | $0.08/kg |
| Plastic | 🟡 Yellow | ✅ | 450-1000 years | $0.15/kg |
| Trash | ⚫ Black | ❌ | Varies | $0.00/kg |

---

## 🤖 How It Works

```
1. Upload Image
       ↓
2. AI Classifies Waste (94-98% accuracy)
       ↓
3. View Disposal Guide
       ↓
4. See Environmental Impact
       ↓
5. Calculate Waste Value (NEW!)
   - Adjust weight slider
   - See monetary + environmental value
   - Estimate bulk sorting value
   - Unlock achievements
```

---

## 📊 Technical Details

### Model Performance
- **Accuracy**: 94-98% on validation set
- **Inference Time**: 50-200ms per image (CPU)
- **Model Size**: 90-100 MB
- **Input Size**: 224 × 224 pixels

### Model Selection
The app auto-loads the best available model in priority order:
1. `waste_fusion_top2.keras` (best)
2. `waste_ensemble_top2.keras` (very good)
3. `waste_model.keras` (good - default)
4. `waste_best_model_InceptionV3.h5` (acceptable)

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Epochs | 10 (initial) + 15 (fine-tune) |
| Batch Size | 32 |
| Optimizer | Adam (lr=0.001) |
| Loss | Categorical Cross-Entropy |
| Data Split | 80% train / 20% val |

---

## 💰 Cost Calculator Details

### Pricing Model
Based on 2024 commodity market rates:
- **Metal**: $0.50/kg (aluminum, copper, steel)
- **Plastic**: $0.15/kg (mixed recyclables)
- **Paper**: $0.08/kg
- **Cardboard**: $0.05/kg
- **Glass**: $0.03/kg
- **Trash**: $0.00/kg (no market value)

### Carbon Credits
- **Price**: $0.30 per kg CO₂ saved
- **Standard**: Verified Carbon Standard (VCS) industry rate
- **Example**: 1kg metal = 0.90 kg CO₂ saved = $0.27 carbon credit

### Gamification
- Select bulk quantity (10-1000 items)
- Assume ~100g per household item
- See total monetary value + environmental impact
- Unlock achievement badge at $50+ value

**Example**: Sorting 100 metal items
- Monetary value: $5.00
- CO₂ saved: 9.0 kg (≈ 1 car removed for 1 day)
- Carbon credits: $2.70
- **Total: $7.70** ✨

---

## 🚀 Deployment

### Streamlit Cloud (Recommended)
1. Push to GitHub
2. Visit https://share.streamlit.io
3. Connect GitHub account
4. Select repository & `app.py`
5. Deploy!

**Live Version**: https://aayushagarwalcnn.streamlit.app

### Local Server
```bash
streamlit run app.py --server.port=8501
```

### Docker (Optional)
```bash
docker build -t wasteai .
docker run -p 8501:8501 wasteai
```

---

## 📁 File Structure

```
wasteai/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Streamlit runtime config
├── README.md                       # This file
├── waste_model.keras               # Main model (default)
├── waste_best_model_InceptionV3.h5 # Alternative model
└── waste_best_model_EnsembleTop2.h5 # Backup model
```

---

## 🛠️ Dependencies

```
streamlit>=1.55.0       # Web framework
tensorflow>=2.21.0      # Deep learning
keras>=3.12.1          # Neural networks
numpy>=2.0.0           # Numerical computing
Pillow>=10.4.0         # Image processing
plotly>=5.24.1         # Interactive charts
h5py>=3.11.0           # Model file format
```

---

## ❓ FAQ

### Q: Why is my prediction low confidence?
A: Poor image quality, bad lighting, or unclear waste. Ensure good image of the waste item.

### Q: Can I use my own model?
A: Yes! Replace the `.keras` or `.h5` files. The app auto-detects.

### Q: How accurate is the cost calculator?
A: Based on real commodity market rates. Prices vary by region and time.

### Q: Can this integrate with actual recycling centers?
A: Yes! The system is designed for API integration with real payment systems.

### Q: What about regional pricing differences?
A: The code supports dynamic pricing. Update `MARKET_PRICES` dict in app.py or integrate with a pricing API.

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 94-98% |
| Inference Speed | 50-200ms |
| Model Size | 90-100 MB |
| Supported Classes | 6 waste types |
| Lines of Code | ~900 (app.py) |
| New Feature | Cost Calculator |

---

## 🎓 Use Cases

### Educational
- Teach waste management economics
- Gamified learning about recycling
- Quantify environmental impact

### Organizational
- School/office waste tracking
- Sustainability ROI calculation
- Engagement & competition

### Environmental
- Proof-of-concept for payment systems
- Integration with recycling facilities
- Real incentive models for proper sorting

---

## 🔐 Security & Privacy

- No images stored on server
- Processing happens locally
- No personal data collected
- Open source - inspect code freely

---

## 📝 License

MIT License - Feel free to use for educational and commercial projects.

---

## 🤝 Contributing

Found a bug? Want to improve? 
1. Fork repository
2. Create feature branch
3. Submit pull request

---

## 👨‍💻 Author

**Aayush Agarwal**  
Smart Waste Segregation System using CNN  
GitHub: [@Aayush-Agarwal007](https://github.com/Aayush-Agarwal007)

---

## 📞 Support

- Check this README
- Review code comments in app.py
- Open an issue on GitHub

---

**Status**: ✅ Production Ready  
**Last Updated**: 2026  
**Version**: 2.0 (with Cost Calculator)
