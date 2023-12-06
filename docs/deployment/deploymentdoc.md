# Despliegue de modelos

## Infraestructura

- **Nombre del modelo:** weights.h5 (este contiene el modelo en vez de los pesos)
- **Plataforma de despliegue:** La plataforma donde se va a desplegar el modelo es FastApi.
- **Requisitos técnicos:** Los requisitos técnicos necesarios para el despliegue son:
  fastapi
  pydantic
  tensorflow>=2.10.0 (versión usada 2.14.0)  
  pillow
  uvicorn
  La versión de Python usada para desarrollar el modelo fue 3.10.12.
  
- **Requisitos de seguridad:** El modelo desarrollado no tiene requisitos de seguridad necesarios para el despliegue.
- 
- **Diagrama de arquitectura:** (imagen que muestra la arquitectura del sistema que se utilizará para desplegar el modelo)

## Código de despliegue
- **Archivo principal:** Main.py
  
- **Rutas de acceso a los archivos:** Los archivos necesarios para el despliegue están disponibles en: https://github.com/S4G0/project_mlds/tree/master/scripts/evaluation_and_deployment
  
## Documentación del despliegue

- **Instrucciones de instalación:** A continuación las instrucciones detalladas para instalar el modelo en la plataforma de despliegue:
- Clonar el Repositorio: Clonar el repositorio que contiene el código del modelo desde el repositorio remoto "git clone https://github.com/S4G0/project_mlds.git".
- Instalar dependencias: Disponibles en el primer punto de infraestructura de este archivo (requisitos técnicos).  
- **Instrucciones de configuración:**-
- Instalar las siguientes librerías¨:
  import requests
  from pydantic import BaseModel

- **Instrucciones de uso:** Ejecutar las siguientes líneas de código (Nota: actualizar el valor de path_test con la imagen a analizar):

    model_url = "https://projectmlds-production.up.railway.app" 
    
    class ApiInput(BaseModel):
        features: str
    
    class ApiOutput(BaseModel):
        forecast: float
    
    path_test='benign_test.jpg'  #Actualizar path_test con la imagen a analizar.
    inp = ApiInput(features=path_test)
    r = requests.post(
        os.path.join(model_url, "predict"),
        json=inp.dict(),
        )
    print(r.json()) 


- **Instrucciones de mantenimiento:** Revisar cada 6 meses si las imagenes están siendo clasificadas correctamente, en caso contrario re-entrenar el modelo.


