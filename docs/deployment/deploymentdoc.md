# Despliegue de modelos

## Infraestructura

- **Nombre del modelo:** Melanoma_model.
- **Plataforma de despliegue:** La plataforma donde se va a desplegar el modelo es FastApi.
- **Requisitos técnicos:** Los requisitos técnicos necesarios para el despliegue son:
  scikit-learn (versión usada 1.2.2)
  tensorflow>=2.10.0 (versión usada 2.14.0)
  fastapi
  uvicorn
  Pillow
  La versión de Python usada para desarrollar el modelo fue 3.10.12.    
- **Requisitos de seguridad:** El modelo desarrollado no tiene requisitos de seguridad necesarios para el despliegue.
- **Diagrama de arquitectura:** (imagen que muestra la arquitectura del sistema que se utilizará para desplegar el modelo)

## Código de despliegue

- **Archivo principal:** Main.py
- **Rutas de acceso a los archivos:** Los archivos necesarios para el despliegue están disponibles en: https://github.com/S4G0/project_mlds/tree/master/scripts/evaluation_and_deployment
- **Variables de entorno:** (lista de variables de entorno necesarias para el despliegue)

## Documentación del despliegue

- **Instrucciones de instalación:** A continuación las instrucciones detalladas para instalar el modelo en la plataforma de despliegue:
- Clonar el Repositorio: Clonar el repositorio que contiene el código del modelo desde el repositorio remoto "git clone https://github.com/S4G0/project_mlds.git".
- Instalar dependencias: Disponibles en el primer punto de infraestructura de este archivo.  
- **Instrucciones de configuración:** Abrir el archivo principal (main.py).
- **Instrucciones de uso:** Ejecutar la aplicación de fastapi (uvicorn app.main:app --reload) y acceder a la interfaz del modelo. 
- **Instrucciones de mantenimiento:** Monitorizar regularmente las actualizaciones de seguridad de FastAPI y otras dependencias. Mejorar y actualizar según sea necesario.


