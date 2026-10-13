# 🔍 GraphSense: Graph Search Relevance

[![CI Pipeline](https://github.com/MSBHOSLE/graph-search-relevance/actions/workflows/ci.yml/badge.svg)](https://github.com/MSBHOSLE/graph-search-relevance/actions)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://python.org)
[![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Topology-blue?logo=networkx&logoColor=white)](https://networkx.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-TF--IDF%20%7C%20SVM-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **Hybrid Graph Search Relevance Engine** combining **Node2Vec graph embeddings** with **TF-IDF text similarity** and **SVM noise filtering** to rank and retrieve multi-relational knowledge entities.

---

## ⚡ Highlights

- **Hybrid Ranking Formulation**: Weighted interpolation $S = \alpha \cdot \text{Sim}_{\text{Graph}} + (1 - \alpha) \cdot \text{Sim}_{\text{Text}}$.
- **Graph Topology Modeling**: Incorporates random-walk neighborhood structural context.
- **Noise Rejection**: Support Vector Machine decision boundary pruning weak semantic associations.

---

## 🚀 Running Tests

```bash
python -m unittest discover tests
```

---

## 👤 Author

**Mahesh Sanjayrao Bhosle**
- **GitHub:** [@MSBHOSLE](https://github.com/MSBHOSLE)
- **LinkedIn:** [Mahesh Bhosle](https://linkedin.com/in/mahesh-bhosle-916991198)
- **Email:** msbhosle.2@gmail.com
- **Education:** M.Sc. Data & Computer Science, Heidelberg University

---

## 📄 License

MIT License - see [LICENSE](LICENSE).
