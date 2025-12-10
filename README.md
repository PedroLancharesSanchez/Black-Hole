# Black-Hole
Black-Hole es un sistema de clasificación de imagenes a gran escala en el espacio bidimensional.

<img width="1917" height="864" alt="image" src="https://github.com/user-attachments/assets/793046aa-469a-4c04-ad0b-a879a877d2df" />

Las imagenes son embedidas por una red neuronal tipo ResNet y posteriormente se comprime el vector extraido a un espacio 2D por el método seleccionado.

<img width="1915" height="854" alt="image" src="https://github.com/user-attachments/assets/2ca2778b-bae9-4514-b3cc-42d843f87834" />

Este espacio bidimensional puede ser clasificado en regiones. De este modo, podemos definir regiones del espacio que definen una clase.

<img width="1919" height="861" alt="image" src="https://github.com/user-attachments/assets/c1390b84-a75d-48eb-965f-dd8b71c359ca" />

Posteriormente, podemos exportar estas etiquetas en formato csv. De este modo se logra clasificar grandes cantidades de imagenes en muy poco tiempo.





