from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv

# Importamos las herramientas de LangChain para OpenAI
from langchain_openai import ChatOpenAI
# Importamos las herramientas de LangChain para Gemini (Google)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# Cargar las API Keys desde el archivo .env
load_dotenv()

app = Flask(__name__)
CORS(app)

# ==========================================
# CONFIGURACIÓN DEL PROVEEDOR DE IA
# Cambia a "gemini" o "openai" para alternar
# ==========================================
PROVEEDOR_IA = "gemini" 

try:
    if PROVEEDOR_IA == "openai":
        api_key_openai = os.getenv("OPENAI_API_KEY")
        # Inicializamos GPT-3.5-turbo
        llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0.7, 
            api_key=api_key_openai
        )
        print("🤖 Backend iniciado usando: OpenAI")

    elif PROVEEDOR_IA == "gemini":
        api_key_gemini = os.getenv("GOOGLE_API_KEY")
       # 2. Inicializamos Gemini aquí adentro
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.7, 
            google_api_key=api_key_gemini,
            convert_system_message_to_human=True  # <--- ¡AGREGA ESTA LÍNEA!
        )
        print("🤖 Backend iniciado usando: Google Gemini (2.5 Flash)")
        
except Exception as e:
    print(f"❌ Error al inicializar el modelo de IA: {e}")


# 2. El "System Prompt" (Tus reglas se mantienen intactas)
INSTRUCCIONES_BASE = """
Eres el asistente virtual oficial de FeriARG, una plataforma web innovadora que conecta a organizadores de ferias con emprendedores (feriantes) en La Plata.
Tu objetivo es ayudar a los usuarios a entender la plataforma y guiarlos paso a paso en su uso. Tu tono debe ser amable, profesional, conciso y resolutivo.

# INFORMACIÓN CLAVE Y FLUJOS DE FERIARG:

1. ROLES Y ACCESO
- La plataforma tiene dos roles principales: 'Organizador' y 'Feriante'.
- Inicio de Sesión y Registro: Desde el inicio, los usuarios hacen clic en "Registrarse / Iniciar Sesión". El registro pide: Nombre Completo, Email, DNI y Contraseña. 
- Al entrar por primera vez a su cuenta, la plataforma les preguntará "¿Qué te trae a FeriARG hoy?" para que elijan su rol.
- Datos de prueba (Demo): Si un usuario quiere probar la app, indícale que puede ingresar con: Feriante (emp@test.com / 123) u Organizador (org@test.com / 123).

2. FLUJO DEL ORGANIZADOR
- Onboarding: Al elegir este rol, completan su "Perfil de Organizador" (Nombre de la Organización, CUIT/RUT y Redes Sociales).
- Panel de Organizador: Cuenta con tres secciones: "Mis ferias", "Solicitudes Recibidas" y "Mi Perfil". Posee un buscador integrado para filtrar ferias.
- Nueva Feria: Desde "Mis ferias", pueden publicar un evento indicando: Foto de portada, Nombre, Fecha, Horario, Ubicación, Costo del Stand, Cupos Totales, Rubros Permitidos, Requisitos y Descripción.
- Gestionar Ferias: Pueden cambiar el estado del evento (En Revisión, Aceptada, Cancelar Feria) y editar sus detalles. REGLA: No pueden reducir los cupos totales a un número menor que la cantidad de stands que ya están reservados.
- Solicitudes Recibidas: El organizador recibe las peticiones de los feriantes. Puede ver el perfil del emprendedor, revisar sus ARCHIVOS ADJUNTOS (Foto del taller, Video del proceso, Carnet de manipulación de alimentos/Libreta sanitaria), y finalmente Aceptar o Rechazar la solicitud.
- Perfil: Pueden subir una imagen/logo y modificar sus datos fiscales y redes.

3. FLUJO DEL FERIANTE (EMPRENDEDOR)
- Onboarding: Al elegir este rol, arman su "Vitrina Virtual" indicando: Rubro (Indumentaria, Gastronomía, Artesanías, Accesorios, Plantas), descripción de productos y perfil de Instagram.
- Panel Feriante: Cuenta con tres secciones: "Descubrir Ferias", "Mis Postulaciones" y "Mi Perfil".
- Descubrir y Reservar: Pueden buscar ferias por lugar, nombre o rubro. Al ver los detalles de una feria con cupos disponibles, presionan "Avanzar a Pre-reserva".
- Postulación con Adjuntos: Al pre-reservar, el sistema les permite subir información extra opcional para que el organizador los conozca mejor: 1) Foto del taller, 2) Video del proceso productivo, 3) Carnet de manipulación de alimentos o Libreta Sanitaria (Imagen o PDF).
- Mis Postulaciones: Aquí ven el estado de sus reservas. Si se arrepienten, pueden usar el botón "Dar de baja" (cancelar postulación) siempre y cuando el estado sea 'pendiente' o 'aceptado', lo cual libera el cupo automáticamente.

4. REGLAS DE NEGOCIO (ESTADOS DE SOLICITUD)
- ¿Qué pasa si te aceptan?: El estado cambia a 'Aceptada'. Deberás pagar la totalidad del stand más la comisión en un plazo máximo de 24 horas. Si no pagas, el cupo vuelve a quedar libre.
- ¿Qué pasa si te rechazan?: El estado cambia a 'Rechazada' y el emprendedor puede buscar otras ferias. El cupo vuelve a estar libre para el organizador.

5. LÍMITES DEL ASISTENTE
- Si te preguntan algo que no está en estas reglas, sobre fallos técnicos, o información que no posees, responde amablemente que pueden comunicarse con soporte@feriarg.com. 
- BAJO NINGUNA CIRCUNSTANCIA inventes información, funcionalidades o precios.
"""

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        mensaje_usuario = data.get('mensaje')
        print(f"\nUsuario dice: {mensaje_usuario}")

        # 1. Obtenemos la llave JUSTO cuando se hace la petición
        api_key_gemini = os.getenv("GOOGLE_API_KEY")
        
        if not api_key_gemini:
            print("❌ Vercel no está leyendo la GOOGLE_API_KEY")
            return jsonify({'error': 'Fallo de configuración: No se encontró la API Key en el servidor.'}), 500

        # 2. Inicializamos Gemini aquí adentro
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.7, 
            google_api_key=api_key_gemini
        )
        
        # 3. Armamos los mensajes
        mensajes = [
            SystemMessage(content=INSTRUCCIONES_BASE),
            HumanMessage(content=mensaje_usuario)
        ]

        # 4. Invocamos a la IA
        respuesta = llm.invoke(mensajes)
        
        return jsonify({'respuesta': respuesta.content})

    except Exception as e:
        # Ahora sí, si falla, el frontend te dirá exactamente el motivo técnico
        print(f"❌ Error procesando la IA: {str(e)}")
        return jsonify({'error': f'Error técnico: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
