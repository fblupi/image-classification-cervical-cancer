| Número de solución | Descripción | Algoritmo y software empleado | Resultado (test) | Resultado (training) | Posición | Fecha y hora |
| ---:| --- | --- | ---:| ---:| ---:|:---:|
| 1 | CNN simple learning from scratch con datos de train básicos | R, mxnet | 1.11456 | 1.0688049 | 671 | 5/6/2017 10:25 | 
| 2 | CNN simple learning from scratch con más datos de train y balanceados | R, mxnet | 1.06771 | 1.033483 | 646 | 7/6/2017 11:17 |
| 3 | CNN simple learning from scratch con más datos de train (no balanceados) a 128x128 durante 10 épocas | Python, Keras | 0.9991 | 0.95566 | 412 | 8/6/2017 13:23 |
| 4 | CNN simple learning from scratch con más datos de train (no balanceados) pero a 32x32 durante 20 épocas | Python, Keras | 0.91442 | 0.9575 | 344 | 8/6/2017 17:32 |
| 5 | CNN simple learning from scratch con más datos de train (no balanceados) a 128x128 durante 50 épocas | Python, Keras | 0.94097 | 0.8224 | 347 | 8/6/2017 18:35 |
| 6 | CNN simple learning from scratch con más datos de train (no balanceados) a 128x128 durante 20 épocas | Python, Keras | 0.91071 | 0.8804 | 342 | 8/6/2017 19:16 |
| 7 | CNN bastante más compleja learning from scratch con más datos de train (no balanceados) a 128x128 durante 15 épocas con un batch size muy grande | Python, Keras | 0.99942 | 1.0444 | 344 | 8/6/2017 22:50 |
| 8 | CNN simple learning from scratch con más datos de train (filtrados y algo más balanceados) a 128x128 durante 10 épocas con un batch size muy pequeño | Python, Keras | 0.87668 | 0.8261 | 282 | 9/6/2017 11:23 |
| 9 | Primer intento con finetuning en R | R, mxnet | 2.24609 | 1.2871 | 282 | 9/6/2017 11:34 |
| 10 | Fine tuning ResNet50 con capa FC de tamaño 1024. Epochs=5, batch size=15 | Python, Keras | 0.91451 | 0.6656 | 288 | 10/6/2017 16:00 |
| 11 | Fine tuning ResNet50 con capa FC de tamaño 512. Epochs=10, batch size=15 | Python, Keras | 0.84681 | 0.7661 | 250 | 10/6/2017 22:00 |
| 12 | Fine tuning ResNet50. Epochs=25, batch size=50, sample per epoch=2000 | Python, Keras | 1.11204 | 251 | 0.8318 | 11/6/2017 9:03 |
| 13 | Fine tuning ResNet50 con capa FC de tamaño 512. Epochs=5, batch size=15 | Python, Keras | 0.89159 | 0.8275 | 251 | 11/6/2017 15:15 |
| 14 | Fine tuning ResNet50 con capa FC de tamaño 512. Epochs=10, batch size=15 | Python, Keras | 0.90005 | 0.7875 | 251 | 12/6/2017 15:15 |
| 15 | CNN simple learning from scratch con más datos de train balanceados a 128x128 durante 20 épocas con un batch size muy pequeño | Python, Keras | 0.99534 | 0.7298 | 255 | 12/6/2017 18:13 |