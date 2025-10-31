# MLflow ile Uçtan Uca Metin Sınıflandırma Pipeline'ı

Bu proje, bir metin sınıflandırma modelini eğitmek, en iyi modeli seçmek, model kayıt defterine (Model Registry) kaydetmek ve bir REST API olarak sunmak için `MLflow` kütüphanesini kullanan uçtan uca bir makine öğrenimi iş akışını (MLOps) göstermektedir.

Proje, 42.000 Türkçe haber metnini kullanarak bir modeli (TF-IDF + Lojistik Regresyon) eğitir ve bu modeli "Staging" ve "Production" aşamalarından geçirerek CI/CD (Sürekli Entegrasyon / Sürekli Teslimat) süreçlerini simüle eder.

##  Temel Özellikler

* **Veri Hazırlama:** Metin verilerinin temizlenmesi (küçük harfe çevirme, noktalama ve sayıların kaldırılması).
* **Özel `pyfunc` Modeli:** `TfidfVectorizer` ve `LogisticRegression` adımlarını tek bir `mlflow.pyfunc.PythonModel` sınıfında paketleme. Bu, model sunucusu tarafından çağrıldığında tüm ön işleme ve tahmin adımlarının otomatik olarak uygulanmasını sağlar.
* **Hiperparametre Optimizasyonu:** `ParameterGrid` kullanarak `MLflow` altında iç içe geçmiş çalıştırmalar (nested runs) ile en iyi model parametrelerini (C, max_features, ngram_range) arama.
* **Model Kaydı (Model Registry):** En iyi F1 skoruna sahip modeli otomatik olarak MLflow Model Registry'e kaydetme.
* **CI/CD Simülasyonu:**
    1.  **Staging (CI):** En iyi modelin "Staging" (Hazırlık) aşamasına otomatik olarak terfi ettirilmesi ve bir kalite eşiğine (F1 > 0.60) göre doğrulanması.
    2.  **Production (CD):** Başarılı bir "Staging" doğrulamasının ardından, "insan onayı" (manuel simülasyon) ile modelin "Production" (Canlı) aşamasına alınması.
* **Model Sunucusu:** "Production" modelini yerel bir REST API olarak (port 5001) sunma.
* **API Testi:** Sunulan modele `requests` kütüphanesi ile örnek veriler göndererek canlı tahminler alma.

##  Kullanılan Teknolojiler

* **MLflow:** Deney takibi, model yönetimi, model kaydı ve model sunucusu için ana araç.
* **Scikit-learn:** `TfidfVectorizer` ve `LogisticRegression` için.
* **Pandas:** Veri işleme için.
* **SHAP:** Model açıklanabilirliği için (Not: Çıktılarda analiz başarısız oldu, ancak kod entegrasyonu mevcuttur).

##  İş Akışı (Pipeline)

Proje, `ml_floww.ipynb` notebook'u üzerinden aşağıdaki adımları sırayla çalıştırır:

1.  **Veri Yükleme ve Hazırlık:** `veri/42bin_haber/news` klasöründeki `.txt` dosyaları okunur, temizlenir ve eğitim/test setlerine ayrılır.
2.  **Özel Model Sınıfı:** `TextClassifierPyfunc` sınıfı, `metin` alanını girdi olarak alıp temizleme, TF-IDF dönüşümü ve sınıflandırma adımlarını yürütecek şekilde tanımlanır.
3.  **Grid Search (Parent Run):** `Haber_Siniflandirma_Custom_Pyfunc_GridSearch_v1` deneyi altında bir "Parent Run" başlatılır.
    * **Eğitim (Child Runs):** Belirlenen `param_grid` içindeki her bir parametre kombinasyonu için bir "Child Run" (iç içe çalışma) başlatılır.
    * Her bir "Child Run" içinde model eğitilir, F1 skoru hesaplanır, parametreler, metrikler ve `pyfunc` modeli MLflow'a loglanır.
    * En yüksek F1 skorunu veren "Child Run" ID'si (`best_run_id`) saklanır.
4.  **Model Kaydı ve Staging:**
    * En iyi model (`best_run_id` kullanılarak) `Haber_Siniflandirici_Pyfunc_Prod` adıyla Model Registry'e kaydedilir.
    * Bu yeni model versiyonu otomatik olarak "Staging" aşamasına geçirilir.
5.  **Staging Doğrulaması (CI Simülasyonu):**
    * "Staging" aşamasındaki model (`models:/Haber_Siniflandirici_Pyfunc_Prod/Staging`) yüklenir.
    * Test seti üzerinde F1 skoru tekrar hesaplanır.
    * Eğer F1 skoru belirlenen eşiği (`validation_f1_threshold = 0.60`) geçerse, model versiyonuna "CI/CD Doğrulaması BAŞARILI" şeklinde bir yorum/etiket eklenir.
6.  **Production'a Terfi (CD Simülasyonu):**
    * "Takım Lideri İncelemesi" simüle edilir.
    * CI adımından gelen yorumlar kontrol edilir.
    * `human_approval = True` (insan onayı) olduğu için model "Production" aşamasına terfi ettirilir.
7.  **Model Sunucusu Başlatma (Terminal):**
    * Kullanıcıya, "Production" modelini sunmak için terminalde çalıştırması gereken komut gösterilir.
8.  **API Testi (Inference):**
    * Yerel sunucuya (Port 5001) test verileri gönderilir ve modelin tahminleri (`predicted_category`) başarılı bir şekilde alınır.

##  Projeyi Çalıştırma

Bu projeyi çalıştırmak için `ml_floww.ipynb` notebook'unu kullanabilirsiniz.

### Gereksinimler
Ayrıca, `veri/42bin_haber/news` klasör yapısının ve `.txt` dosyalarının notebook'un erişebileceği bir konumda olması gerekmektedir.

### Adım Adım Çalıştırma

1.  **MLflow UI'ı Başlatın (Önerilir):**
    Notebook'u çalıştırmadan önce, ayrı bir terminal açın ve `mlruns` klasörünün oluşturulacağı dizinde şu komutu çalıştırarak MLflow arayüzünü başlatın:
    ```bash
    mlflow ui
    ```
    Bu arayüzü `http://127.0.0.1:5000` adresinden takip edebilirsiniz.

2.  **Notebook'u Çalıştırın:**
    `ml_floww.ipynb` notebook'undaki tüm hücreleri **sırayla** çalıştırın.
    * Hücre 1-5, modeli eğitecek, en iyisini seçecek ve "Production" aşamasına kadar getirecektir.

3.  **Model Sunucusunu Başlatın (Hücre 6 Talimatı):**
    Notebook'un 6. hücresindeki talimatları izleyin. **Ayrı bir terminal** açın, `mlruns` klasörünün olduğu dizine gidin ve şu komutu çalıştırın:
    ```bash
    mlflow models serve -m "models:/Haber_Siniflandirici_Pyfunc_Prod/Production" --port 5001 --no-conda
    ```
    Sunucu `Listening at: http://127.0.0.1:5001` mesajını verdiğinde hazır demektir.

4.  **API'yi Test Edin (Hücre 7):**
    Notebook'a geri dönün ve son hücreyi (Bölüm 2) çalıştırın. Bu hücre, çalışan sunucuya istek atacak ve tahminleri alacaktır.

##  Örnek API İsteği (Python)

Aşağıdaki kod, çalışan model sunucusuna (Port 5001) nasıl istek atılacağını göstermektedir.

```python
import requests
import json

test_data_json = {
    "dataframe_split": {
        "columns": ["metin"],
        "data": [
            ["Galatasaray şampiyonluk maçına çıkıyor, tüm biletler satıldı"],
            ["Enflasyon rakamları açıklandı, merkez bankası faiz kararı bekleniyor"],
            ["Yapay zeka modelleri metin özetlemede çığır açtı"]
        ]
    }}

url = "[http://127.0.0.1:5001/invocations](http://127.0.0.1:5001/invocations)"

response = requests.post(
    url,
    data=json.dumps(test_data_json),
    headers={"Content-Type": "application/json"}
)

print(json.dumps(response.json(), indent=2, ensure_ascii=False))
```

Başlamadan önce gerekli kütüphanelerin kurulu olduğundan emin olun:

```bash
pip install mlflow pandas scikit-learn numpy shap plotly matplotlib requests
```

# Örnek Çıktı
```json
{
  "predictions": [
    {
      "input_text": "Galatasaray şampiyonluk maçına çıkıyor, tüm biletler satıldı",
      "predicted_category": "spor"
    },
    {
      "input_text": "Enflasyon rakamları açıklandı, merkez bankası faiz kararı bekleniyor",
      "predicted_category": "genel"
    },
    {
      "input_text": "Yapay zeka modelleri metin özetlemede çığır açtı",
      "predicted_category": "guncel"
    }
  ]
}
```
