# Demo Interactiva

Una interfaz de navegador para los tres proyectos. Cada panel llama al mismo
código de librería que usan las herramientas de línea de comandos, así que lo que
ves es el pipeline real corriendo en tu máquina, no una grabación de uno.

## Cómo lanzarla

Desde la raíz del repositorio, con los tres proyectos instalados:

```bash
pip install -e structure-from-motion -e gaussian-splatting -e gradient-domain
pip install -r interactive-demo/requirements.txt

python interactive-demo/app.py
```

Se abre en `http://127.0.0.1:7860`. Añade `--share` para una URL pública
temporal, `--port` para moverla y `--no-browser` para que no abra una ventana.

El panel de Gaussian splatting necesita un checkpoint entrenado en
`gaussian-splatting/docs/model.npz`. Si falta, ese panel lo dice y muestra el
comando que lo produce; los otros dos funcionan igualmente.

## Qué hace cada panel

**Structure from motion.** Genera una escena sintética con verdad conocida, la
reconstruye e informa del error contra esa verdad. Los dos controles que importan
son el ruido de medida y la fracción de outliers groseros: sube cualquiera de los
dos y puedes ver degradarse en un orden concreto al solver de cinco puntos, los
estimadores robustos y el bundle adjuster. La vista 3D se puede orbitar, así que
la trayectoria de cámaras y la estructura recuperada se inspeccionan desde
cualquier ángulo. Una pasada tarda unos segundos, así que vive detrás de un botón.

**Gaussian splatting.** Renderiza el modelo entrenado desde cualquier punto de
vista de la órbita y traza por rayos la misma cámara al lado. Ninguno de los dos
puntos de vista estuvo en el conjunto de entrenamiento, así que la diferencia
entre los dos paneles es generalización y no ajuste. Lleva la elevación a sus
extremos, donde no llegó ninguna cámara de entrenamiento, y la representación
empieza a descomponerse como lo hacen los modelos de splatting. Cada render tarda
una fracción de segundo, así que la vista sigue a los sliders.

**Dominio del gradiente.** Cuatro operaciones repartidas en cuatro sub-pestañas:
clonado sin costura, mapeo tonal, aplanado de textura y contraste local. Cada
resolución son decenas de milisegundos, así que el resultado sigue a los controles
directamente. Dos cosas merecen un barrido: el exponente del mapeo tonal, cuyo
convenio va al revés del que uno esperaría, y la elección de solver en el dominio
rectángulo, donde los conteos de iteraciones del gradiente conjugado y de las
variantes multigrid se imprimen uno al lado del otro.

## Estructura

```
interactive-demo/
  app.py                   monta las tres pestañas
  common.py                conversión de imágenes, tablas de métricas, colocación de cámara
  reconstruction_panel.py  structure from motion, con una vista 3D orbitable
  splatting_panel.py       vistas nuevas contra una referencia trazada por rayos
  gradient_panel.py        las cuatro operaciones de dominio del gradiente
```

## Alcance

Esto es una superficie de demostración, no parte de ninguno de los tres paquetes:
nada de aquí lo importan ellos, y nada de esto está cubierto por sus tests. Es
deliberadamente monousuario y de un solo proceso, así que dos personas apuntando
el navegador a la misma instancia harán cola en la GPU.
