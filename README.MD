**<h1> Detección de objetos con Yolov5 </h1>**

En este tutorial, nos enfocamos en la detección de autos, utilizando Yolov5x (La arquitectura de Yolov5 más compleja).

**Yolov5**: Es una familia de arquitecturas y modelos de deteccion de objetos entrenados previamente en el conjunto de datos COCO.

***<h2>Procedimiento: </h2>***

**<h3>1. Extracción del Conjunto de imágenes</h3>**

Descargamos la extensión de [Download All Images](https://chrome.google.com/webstore/detail/download-all-images/ifipmflagepipjokmbdecpmjbibjnakm), esto nos permitirá descargar un lote de imágenes en el buscador de google. 

<img src='images_readme/img1.png'  width='100%'>
<img src='images_readme/img5.png'  width='100%'>
<p align='center'><b>Seleccionamos imágenes específicando el en la extensión que descargaremos imágenes mayores a 160x160 pixeles, conforme hagamos scroll en el buscador, la extensión me descargará todas esas imágenes</b></p>
<div style = "text-align:center;">
<img src='images_readme/img2.png'  width='50%'>
<p><b>Hacemos click en la extensión, y me descargará un archivo .rar con todas las imágenes</b></p>
</div>

<div style = "text-align:center;">
<img src='images_readme/img3.png'  width='50%'>
<p><b>Creamos una carpeta data, que contendrá dos carpetas <b>images</b> y <b>labels</b>, cada carpeta tendra las carpetas train(imágenes etiquetadas para el entrenamiento) y val(imágenes etiquetadas para la validación), en cuestión de la carpeta <b>labels</b> contendrá las etiquetas (cajas delimitadoras de cada imagen) referente a la carpeta images</p>
</div>

**<h3>2. División Conjunto de imágenes</h3>**
Descomprimos el archivo de imágenes y procedemos a eliminar imágenes que no sea útiles, fotos de autos incompletas, dibujos de autos, etc.
<div style = "text-align:center;">
<img src='images_readme/img4.png'  width='100%'>
<p><b>Luego dividimos estas imágenes en 70% (train) y 30% (val)</b> en nuestro caso son 106 train y 46 val, y son guardadas en las carpetas images/train y images/val </p>
</div>

**<h3>3. Etiquetado del Conjunto de imágenes</h3>**
Utilizaremos [MakeSense](https://www.makesense.ai/) para hacer el etiquetado de autos en las imágenes

<div style = "text-align:center;">
<img src='images_readme/img6.png'  width='100%'>
<p><b>Arrastramos por carpeta, empezamos con train </b> y al finalizar hacemos lo mismo con val</p>
</div>

Luego especificamos el nombre de la etiqueta a detectar
<div style = "text-align:center;">
<img src='images_readme/img7.png'  width='100%'>
</div>

Empezamos a definir el cuadro delimitador del objeto a detectar por cada imagen
<div style = "text-align:center;">
<img src='images_readme/img8.png'  width='100%'>
</div>

Una vez finalizado, exportamos las etiquetas en el formato adecuado para yolo.
<div style = "text-align:center;">
<img src='images_readme/img9.png'  width='100%'>
</div>

Luego se nos descarga un archivo .rar, el cual descomprimos mostrandonos por cada imagen su respectivos cuadros etiquetados de autos
<div style = "text-align:center;">
<img src='images_readme/img10.png'  width='100%'>
</div>

**<h3>4. Entrenamiento de imágenes con Yolov5x</h3>**

Ahora nos dirigimos al [link](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) de entrenamiento especificado por yolov5, y abrimos el `YOLOv5 Custom Training Notebook`, Este notebook nos da detalles del entrenamiento que debemos seguir paso a paso.
<div style = "text-align:center;">
<img src='images_readme/img11.png'  width='100%'>
<p><b>Antes debemos descargar el archivo coco128.yaml, en donde especificaremos la ruta de nuestros datos dentro de colab y el nombre de clase</b> </p>
</div>
<div style = "text-align:center;">
<img src='images_readme/img13.png'  width='100%'>
<p><b>Especificando ruta de datos y clases</b> </p>
</div>

Comprimimos a un archivo rar nuestros datos, y arrastramos en la carpeta de colab, para después descomprimilar dentro y tener nuestros datos en ese entorno
<div style = "text-align:center;">
<img src='images_readme/img12.png'  width='50%'>
</div>

Luego de modificar el coco.yaml, lo renombramos a custom.yaml, y enviamos a la misma ubicación donde la descargamos
<div style = "text-align:center;">
<img src='images_readme/img14.png'  width='50%'>
</div>

Luego configuramos el entrenamiento, estableciendo el input size, mini batch, epochs, url del curstom.yaml y el tipo de modelo de yolov5
<div style = "text-align:center;">
<img src='images_readme/img16.png'  width='100%'>
</div>

<div style = "text-align:center;">
<img src='images_readme/img15.png'  width='100%'>
<p><b>Alcanzandose en 20 epochs una precisión del 97%</b> </p>
</div>

**<h3>5. Descargar pesos de entrenamiento</h3>**
Una vez finalizado, podemos descargar nuestros pesos, para que posteriormente podamos probarlos en nuestras propias inferencias, el modelo descargado se puede descargar en este [enlace](https://drive.google.com/file/d/14SqH1cQtLdpAq0rqtJerlvL9yyUzG3uN/view?usp=share_link).
<div style = "text-align:center;">
<img src='images_readme/img17.png'  width='100%'>
<p><b>Archivo del tipo pytorch</b> </p>
</div>

**<h3>6. Detecciones personalizadas</h3>**
Según la documentación de yolov5, para poder instalarlo en nuestro entorno debemos instalar los requerimientos que se especifican [aquí](https://docs.ultralytics.com/tutorials/pytorch-hub/)
<div style = "text-align:center;">
<img src='images_readme/img18.png'  width='100%'>
<p><b>Instalaciones necesarias</b> </p>
</div>


**<h3>7. Cargando modelo personalizado</h3>**
Ejecutamos el código del archivo detect.py, y se nos abrirá la camará que tomara frames para inferir si contiene autos a detectar
<div style = "text-align:center;">
<img src='images_readme/img19.jpg'  width='100%'>
<p><b>Deteccion de dos imagenes de autos</b> </p>
</div>



