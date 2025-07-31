# Machine Learning en Finanzas

**Universidad Torcuato Di Tella**

**Profesor: Lionel Modi, CFA**
Email: [lionel.modi@utdt.edu](mailto:lionel.modi@utdt.edu)

**Alumna: Paola Nuñez**
Email: [pnunezherrero@utdt.edu](mailto:pnunezherrero@utdt.edu)


<br/>

## Detalle de carpetas

`.vscode`

Contiene la configuración de VS Code para debugging.

`addit_notes`

Contiene los resumenes de documentación utilizados a lo largo del curso

`clases`

Contiene los handouts y trabajos realizados en clase.


`data`

Contiene los datasets utilizados en clase


`mlfin`

Paquete de funciones que utilizaremos en el curso.


`ps`

Carpeta donde alojar los Problem Sets Resueltos.

<br/>

## Detalles de archivos

`readme.md`

Este archivo. (Formato Markdown)


`pyproject.toml`
Incluye dependencias y metadatos del proyecto en el formato oficial de configuración PEP 518, 621. 

`.flake8`
Incluye customización de la librería flake8 para code guidelines checking

`.gitignore`
Para no versionar ciertos archivos (principalmente claves y entornos virtuales).

</br>

## Creando un entorno de Anaconda dedicado

**Windows:** Ejecutar las siguientes líneas en *Anaconda Powershell Prompt*

**Linux o Mac:** Ejecutar las siguientes líneas en una terminal.

Correr los siguientes comandos dentro de la capeta donde resida `requirements-conda.txt`

```
conda activate base

// (tensorflow -> 3.12 | PyTorch -> 3.13)
conda create -n mlfin-313 python=3.13 -y
conda activate mlfin-313

conda config --env --add channels conda-forge
conda config --env --add channels nvidia

conda update --all -y

conda install --file requirements-conda.txt -y


// tensorflow (referencia -> https://www.tensorflow.org/install/pip)
(NVIDIA)  pip install tensorflow[and-cuda]
(CPU)     pip install tensorflow
(Apple M) pip install tensorflow tensorflow-metal (referencia -> https://developer.apple.com/metal/tensorflow-plugin)


// pytorch (referencia -> https://pytorch.org/get-started/locally)
(NVIDIA)        pip install torch --index-url https://download.pytorch.org/whl/cu128
(CPU)           pip install torch --index-url https://download.pytorch.org/whl/cpu
(Mac / Apple M) pip install torch
```

</br>

## Creando un entorno virtual con uv

### Opción 1: PowerShell

1. Instalar uv en el entorno global y crear el entorno virtual:
```powershell
pip install uv
uv venv
```

2. Activar el entorno virtual:
```powershell
.venv\Scripts\activate
```

3. Instalar el proyecto en modo editable:
```powershell
uv pip install -e .
```

4. Comandos para gestionar dependencias:
```powershell
# Ver dependencias actuales
uv pip list

# Ver resolución de dependencias
uv pip compile pyproject.toml

# Agregar nueva dependencia
uv add package_name

# Remover dependencia
uv remove package_name

# Agregar dependencia de desarrollo
uv add --dev pytest

# Agregar dependencia opcional
uv add --optional visualization seaborn

# Sincronizar entorno con pyproject.toml (reinstala según el archivo)
uv pip sync

# Actualizar todas las dependencias
uv pip upgrade --all
```

## Check de PEP standard con flake8

Una vez instaladas las dependencias, simplemente usar el comando `flake8`
seguido del documento que queremos que revise. Por otro lado, el archivo
de extensión .flake8 se usa para configurar algunos estándares que se ignoran y otros que se modifican. 

Por ahora, la única modificación que se realiza sistemáticamente de los estándares es la longitud de los renglones (se extienden todos los límites a 100 carácteres).

