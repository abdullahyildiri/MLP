# Yapay Sinir Ağları ile Banknot Doğrulama

## Giriş
Bu çalışma, **Banknot Doğrulama Veri Seti** kullanılarak farklı yapay sinir ağı modellerinin performanslarını karşılaştırmayı amaçlamaktadır. Sinir ağı tabanlı yöntemlerin sahte ve gerçek banknotları ayırt etme başarısı değerlendirilmiştir. Üç farklı model uygulanmıştır:

1. **2 Katmanlı Yapay Sinir Ağı (2_Layer.py)**
2. **3 Katmanlı Yapay Sinir Ağı (3_Layer.py)**
3. **scikit-learn MLPClassifier (scikit_learn.py)**

Bu modeller doğruluk, kesinlik, geri çağırma ve F1 skoru gibi metriklerle karşılaştırılmıştır.

## Yöntem

### Veri Seti
**BankNote_Authentication.csv** veri seti kullanılmıştır. Veri seti aşağıdaki özelliklerden oluşmaktadır:
- Varyans
- Çarpıklık
- Eğiklik
- Entropi
- Etiket (Sahte = 0, Gerçek = 1)

Veri %80 eğitim, %20 test olarak ayrılmıştır.

### Modeller
1. **2 Katmanlı Yapay Sinir Ağı (2_Layer.py):**
   - **Katman Sayısı:** 2
   - **Aktivasyon Fonksiyonu:** Tanh ve Sigmoid
   - **Öğrenme Oranı:** 0.01
   - **İterasyon Sayısı:** 800
   - **En iyi yapılandırma:** 3 gizli nöron, 800 iterasyon
   - **En iyi doğruluk:** 0.9818
   
2. **3 Katmanlı Yapay Sinir Ağı (3_Layer.py):**
   - **Katman Sayısı:** 3
   - **Aktivasyon Fonksiyonu:** ReLU ve Sigmoid
   - **Öğrenme Oranı:** 0.03
   - **İterasyon Sayısı:** 1000
   - **En iyi yapılandırma:** 4 gizli nöron, 1000 iterasyon
   - **En iyi doğruluk:** 0.9964
   
3. **MLPClassifier (scikit_learn.py):**
   - **Katman Sayısı:** 1 gizli katman (6 nöron)
   - **Aktivasyon Fonksiyonu:** ReLU
   - **Optimizasyon:** Stochastic Gradient Descent (SGD)
   - **Öğrenme Oranı:** 0.003
   - **İterasyon Sayısı:** 800

## Sonuçlar
Aşağıdaki tabloda her modelin performans karşılaştırması verilmiştir:

| Model | Doğruluk | Kesinlik | Geri Çağırma | F1 Skoru |
|--------|----------|-----------|--------------|---------|
| **2 Katmanlı Sinir Ağı** | 0.9818 | 0.9865 | **0.9881** | **0.9873** |
| **3 Katmanlı Sinir Ağı** | **0.9964** | **0.9890** | 0.9814 | 0.9852 |
| **MLPClassifier** | 0.9832 | 0.9835 | 0.9828 | 0.9831 |

## Tartışma
- **3 Katmanlı Sinir Ağı**, en yüksek doğruluk oranına ulaşmıştır (%99.64). Daha derin bir model olması nedeniyle daha iyi genelleme yapabilmiştir.
- **2 Katmanlı Sinir Ağı**, daha basit bir yapı ile yüksek doğruluk sağlamış olsa da en iyi yapılandırmada %98.18 doğruluğa ulaşmıştır.
- **MLPClassifier**, daha kısa sürede eğitilmesine rağmen diğer modellere kıyasla daha düşük doğruluk göstermiştir.

Bu sonuçlar, **katman ve nöron sayısının doğru seçilmesinin model başarısını önemli ölçüde etkilediğini** göstermektedir. Optimizasyon teknikleriyle performans daha da artırılabilir.

## Referanslar
- Banknote Authentication Data Set: https://archive.ics.uci.edu/ml/datasets/banknote+authentication
- Scikit-learn MLPClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

