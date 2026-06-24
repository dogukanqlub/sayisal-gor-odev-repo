# Deepfake Dedektörü Bozulma Dayanıklılığı — Bildiri Bölümleri (Güncel)

Bu dosyayı Word bildirine (`Deepfake_Bozulma_Dayaniklilik_Bildirisi.docx`) kopyalayabilirsiniz.

---

## 3. Yöntem

### 3.1 Deney Kategorileri

Bozulmalar üç ana grupta incelenmiştir:

| Kategori | Amaç | Örnekler |
|----------|------|----------|
| **Sıkıştırma / kanal gürültüsü** | Dağıtım kanalı simülasyonu | JPEG (q10–90), Gaussian blur, additive noise |
| **Post-processing** | Sosyal medya düzenleme / yeniden paylaşım | Resize (down-up), gamma, kontrast, keskinleştirme, median filtre, doygunluk |
| **Adversarial** | Kasıtlı evasion saldırısı | Label-aware FGSM (ε = 4, 8, 16, 32 piksel) |

**Post-processing** bozulmaları, deepfake içeriğin platformlarda yeniden boyutlandırılması, renk/gamma düzeltmesi veya artefakt gizleme amaçlı median filtre uygulanması gibi gerçek dünya senaryolarını modellemektedir.

**Adversarial** saldırılar, saldırganın dedektörü yanlış sınıflandırmaya zorladığı senaryoyu temsil eder. FGSM, her test karesinin gerçek etiketine göre Cross-Entropy kaybını maksimize edecek yönde pertürbasyon ekler.

### 3.2 Değerlendirme Protokolü

- Model: FaceForensics++ c23 üzerinde eğitilmiş Xception (full image, `full_c23.p`)
- Test: 60 kare (30 real + 30 fake), video bazlı split (frame sızıntısı yok)
- Toplam koşul sayısı: **37** (1 baseline + 14 sıkıştırma/gürültü + 18 post-processing + 4 FGSM)
- Metrikler: Accuracy, AUC, F1, Precision, Recall

---

## 4. Bulgular

### 4.1 Sıkıştırma ve kanal gürültüsü

| Koşul | AUC | Accuracy | Özet |
|-------|-----|----------|------|
| Baseline (c23) | 1.00 | 1.00 | Referans |
| JPEG q10–q90 | 0.98–1.00 | 0.97–1.00 | JPEG'e dayanıklı |
| Blur k3–k7 | 1.00 | 0.93–1.00 | Hafif blur tolere edilir |
| Blur k11 | 0.95 | 0.85 | Belirgin düşüş |
| Noise σ5–σ10 | 1.00 | 0.98–1.00 | Düşük gürültüde etkisiz |
| Noise σ20 | 0.96 | 0.85 | Orta düşüş |
| **Noise σ40** | **0.66** | **0.52** | **Kritik zayıflık** (fake recall %10) |

**Yorum:** JPEG sıkıştırma ve orta düzey blur, c23 üzerinde eğitilmiş modele göre ek risk oluşturmaz. Yüksek Gaussian gürültü (σ=40) ise sahte karelerin %90'ının "gerçek" sanılmasına yol açar; en zayıf kanal bozulması budur.

### 4.2 Post-processing

| Koşul | AUC | Accuracy |
|-------|-----|----------|
| Resize 0.75 / 0.5 | 1.00 | 0.97–1.00 |
| **Resize 0.25** | **0.93** | 0.83 |
| Gamma 0.6–1.4 | 1.00 | 0.97–1.00 |
| Kontrast 0.7–1.3 | 1.00 | 1.00 |
| Sharpen 0.5–2.0 | 1.00 | 0.95–1.00 |
| Median k3–k5 | 1.00 | 0.93–1.00 |
| **Median k7** | **0.88** | 0.80 |
| Saturation 0.5–1.5 | 1.00 | 0.98–1.00 |

**Yorum:** Gamma, kontrast ve doygunluk değişimlerine model güçlü direnç gösterir. Agresif down-up resize (0.25) ve güçlü median filtre (k=7) artefaktları sildiği için AUC düşer; bu, sosyal medyada "yumuşatılmış" deepfake içeriğin tespitini zorlaştırabileceğini gösterir.

### 4.3 Adversarial (FGSM)

| ε (piksel) | AUC | Accuracy | Fake recall |
|------------|-----|----------|-------------|
| 4 | 0.01 | 0.02 | %3 |
| 8 | 0.07 | 0.27 | %7 |
| 16 | 0.35 | 0.42 | %3 |
| 32 | 0.54 | 0.48 | %0 |

**Yorum:** Düşük ve orta ε değerlerinde (4–16) dedektör neredeyse tamamen devre dışı kalır. ε=32'de AUC kısmen yükselir; bu, aşırı pertürbasyonun sınıflandırıcıyı rastgele tahmine itmesinden kaynaklanabilir. Genel olarak adversarial saldırı, tüm doğal bozulma türlerinden daha tehlikelidir.

### 4.4 En zayıf koşullar (özet)

| Tür | En zor koşul | AUC |
|-----|--------------|-----|
| Kanal gürültüsü | noise σ40 | 0.66 |
| Post-processing | median k7 | 0.88 |
| Adversarial | FGSM ε=4 | 0.01 |

---

## 5. Önerilen Çözümler (Training-Free)

Ek model eğitimi gerektirmeyen üç inference-time yöntem önerilmiş ve kritik boşulma koşullarında test edilmiştir. Sonuçlar: `results/tables/solution_comparison.csv`, grafik: `docs/assets/figures/solution_comparison.png`.

### 5.1 TTA (Test-Time Augmentation)

Dört görünümün `fake` olasılık ortalaması: orijinal, JPEG q=70, blur k=3, bilateral filtre.

### 5.2 Temporal Ensemble

Aynı videoya ait kare skorları ortalanır (test setinde 6 video × 10 kare). **Not:** Bu yöntem video düzeyinde değerlendirilir; diğerleri kare düzeyindedir (60 örnek).

### 5.3 Frequency Fusion

FFT yüksek frekans enerji oranı + CNN skoru birleştirilir. Ağırlık `w=0.55`, validation split üzerinde AUC ile kalibre edilmiştir:

```
P_fake = w · P_CNN + (1−w) · P_HF
```

### 5.4 Deneysel Sonuçlar (AUC)

| Koşul | Baseline | TTA | Temporal | Freq fusion |
|-------|----------|-----|----------|-------------|
| Temiz (baseline) | 1.00 | 1.00 | 1.00 | 1.00 |
| Noise σ=40 | 0.66 | 0.68 | **0.78** | 0.66 |
| Blur k=11 | 0.95 | 0.93 | **1.00** | 0.95 |
| Resize 0.25 | 0.93 | 0.92 | 0.89 | 0.93 |
| FGSM ε=16 | 0.35 | 0.07 | 0.11 | 0.35 |

**Fake recall (noise σ=40):** Baseline %10 → Freq fusion **%47** → TTA %10 → Temporal %0

### 5.5 Yorum

- **Temporal ensemble**, gürültü (AUC 0.66→0.78) ve blur (0.95→1.00) koşullarında en güçlü AUC iyileşmesini sağlar; video düzeyinde gürültü ortalaması sinyali stabilize eder.
- **Freq fusion**, gürültüde fake recall'u %10'dan **%47'ye** çıkarır; AUC değişmez — eşik altı sahte tespitinde pratik fayda sağlar.
- **TTA** gürültüde marjinal AUC kazancı verir (+0.02); FGSM'de zararlıdır (0.35→0.07).
- **Hiçbir training-free yöntem FGSM'e karşı yeterli değildir** (en iyi temporal AUC 0.11).

**Paper cümlesi:**
> Training-free çözümlerden temporal ensemble gürültü ve blur bozulmalarına karşı en etkili iyileşmeyi sunar; frequency fusion ise gürültü altında sahte tespit recall'unu anlamlı biçimde artırır. Adversarial saldırılara karşı kalıcı çözüm model eğitimi aşamasında adversarial fine-tuning gerektirir (bu çalışmada kapsam dışı).

### 5.6 Gelecek Çalışma

**Adversarial eğitim:** FGSM + JPEG + blur ile fine-tuning; FGSM ε=16'da AUC > 0.35 hedefi. Tahmini süre: 4–8 saat (CPU eğitim).

---

## 6. Sonuç (ek paragraf)

FaceForensics++ c23 üzerinde eğitilmiş Xception dedektörü, JPEG ve çoğu post-processing bozulmasına dayanıklıdır; ancak yüksek Gaussian gürültü (AUC 0.66), güçlü median filtre (AUC 0.88) ve FGSM saldırıları (AUC 0.01–0.35) altında ciddi performans kaybı yaşar. Test edilen training-free çözümlerden temporal ensemble ve frequency fusion gürültü/blur senaryolarında ölçülebilir iyileşme sağlar; adversarial tehditlere karşı yetersiz kalır.

---

## Kaynakça Ekleri

- Rossler et al., "FaceForensics++", ICCV 2019
- Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015 (FGSM)
- Carlini & Wagner, "Adversarial Examples Are Not Easily Detected", 2017
- Afchar et al., "MesoNet", WIFS 2018

---

## Ek: Kullanılacak görseller

| Dosya | Açıklama |
|-------|----------|
| `docs/assets/figures/progression_fgsm.png` | FGSM görsel ilerlemesi |
| `docs/assets/figures/progression_resize.png` | Resize post-processing |
| `docs/assets/figures/progression_gamma.png` | Gamma post-processing |
| `docs/assets/figures/worst_degradations.png` | En zor 4 koşul karşılaştırması |
| `results/figures/adversarial_fgsm_auc.png` | FGSM ε vs AUC |
| `results/figures/category_heatmap.png` | Tüm kategoriler heatmap |
| `docs/assets/figures/solution_comparison.png` | 4 çözüm yöntemi karşılaştırması (Bölüm 5) |
| `results/figures/worst_case_auc.png` | Kategori bazlı worst-case |
