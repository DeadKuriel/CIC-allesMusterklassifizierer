# Matrices de confusión para demostraciones

`amk confusion-demo` genera una matriz binaria para documentos o diapositivas sin cargar datos,
entrenar modelos ni realizar una clasificación.

Por defecto crea `confusion_demo.png` con esta disposición:

```text
TP | FN
FP | TN
```

```bash
amk confusion-demo
```

Para usar valores manuales, deben proporcionarse en el mismo orden:

```bash
amk confusion-demo --values 50 10 5 35 --output matriz.png
```

Ejemplo personalizado para una diapositiva:

```bash
amk confusion-demo \
  --values 50 10 5 35 \
  --color Blues \
  --title "Ejemplo de clasificación" \
  --class-names Positivo Negativo \
  --font-size 24 \
  --dpi 300 \
  --output matriz.svg
```

Opciones disponibles:

- `--values TP FN FP TN`: muestra cantidades manuales; si se omite, muestra las siglas.
- `--color`: nombre de cualquier paleta instalada de matplotlib, como `Reds`, `Blues` o `Greens`.
- `--title`: título de la figura.
- `--class-names POSITIVO NEGATIVO`: nombres de las dos clases.
- `--output`: ruta `.png`, `.svg` o `.pdf`.
- `--dpi`: resolución, predeterminada en 300.
- `--size ANCHO ALTO`: tamaño de la figura en pulgadas.
- `--font-size`: tamaño del texto central.
- `--text-color` y `--background-color`: colores del texto y fondo.
- `--hide-axis-labels`: oculta “Predicción” y “Valor real”.
- `--show-colorbar`: muestra la escala de color.
- `--transparent`: guarda un fondo transparente.
- `--language es|en`: idioma de los ejes.
