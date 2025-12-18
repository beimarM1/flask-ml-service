# main.py
from fastapi import FastAPI
from sqlalchemy import create_engine
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware  # 🔥 importa el middleware CORS

import numpy as np

app = FastAPI(title="Microservicio de Recomendaciones (Item, User, Hybrid + Info)")
# ==================================================
# 🌍 Habilitar CORS
# ==================================================
origins = [
    "https://backend-ecommerce-production-0ef1.up.railway.app",   
    "https://shopealo-ecomerce.netlify.app/",      
    "*"                      
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],         # permite GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],         # permite cualquier encabezado
)

# ==================================================
# 🔗 Conexión a base de datos y lógica
# ==================================================
DB_URL="postgresql://postgres:ZSfvJIURxqyHJOcSPgBZrjciAlevyqVi@tramway.proxy.rlwy.net:36634/railway"
engine = create_engine(DB_URL)

# Variables globales
matriz_usuario_producto = None
modelo_item = None
modelo_usuario = None
productos_ids = []
usuarios_ids = []
productos_info = None  # para nombre, precio, imagen

# ENTRENAMIENTO GENERAL hhh

def entrenar_modelos():
    global matriz_usuario_producto, modelo_item, modelo_usuario, productos_ids, usuarios_ids, productos_info

    print("📡 Cargando datos de la base de datos... - main.py:47")
    query = """
    SELECT o."usuarioId" AS usuario_id, op."productId" AS producto_id
    FROM "order" AS o
    JOIN order_product AS op ON o.id = op."orderId";
    """
    df = pd.read_sql(query, engine)
    if df.empty:
        print("Sin datos suficientes para entrenar. - main.py:55")
        return

    # Matriz usuario-producto
    matriz_usuario_producto = df.pivot_table(
        index="usuario_id", columns="producto_id", aggfunc=len, fill_value=0
    )

    usuarios_ids = matriz_usuario_producto.index.tolist()
    productos_ids = matriz_usuario_producto.columns.tolist()

    # Similitudes
    modelo_item = cosine_similarity(matriz_usuario_producto.T)
    modelo_usuario = cosine_similarity(matriz_usuario_producto)

    # Info de productos
    productos_info = pd.read_sql("""
        SELECT id, name, price, "urlImage"
        FROM product;
    """, engine)

    print(f"Modelos entrenados con {len(productos_ids)} productos y {len(usuarios_ids)} usuarios.")


entrenar_modelos()


# Función auxiliar para obtener info de productos
def info_productos(ids):
    if productos_info is None or len(ids) == 0:
        return []
    df = productos_info[productos_info["id"].isin(ids)]
    return df.to_dict(orient="records")


# =========================
# ITEM-BASED
# =========================
@app.get("/recomendaciones/item/{producto_id}")
def recomendar_item_based(producto_id: int, cantidad: int = 5):
    if producto_id not in productos_ids:
        return {"error": f"Producto {producto_id} no encontrado."}

    idx = productos_ids.index(producto_id)
    similitudes = modelo_item[idx]
    indices = np.argsort(similitudes)[::-1][1:cantidad + 1]
    recomendados = [int(productos_ids[i]) for i in indices]

    base = info_productos([producto_id])[0] if producto_id in productos_info["id"].values else {"id": producto_id}
    recomendados_info = info_productos(recomendados)

    return {
        "modelo": "item-based",
        "producto_base": base,
        "recomendados": recomendados_info
    }


# =========================
# USER-BASED
# =========================
@app.get("/recomendaciones/usuario/{usuario_id}")
def recomendar_user_based(usuario_id: int, cantidad: int = 5):
    if usuario_id not in usuarios_ids:
        return {"error": f"Usuario {usuario_id} no encontrado."}

    idx_usuario = usuarios_ids.index(usuario_id)
    similitudes = modelo_usuario[idx_usuario]
    usuarios_similares = np.argsort(similitudes)[::-1][1:6]

    productos_usuario = set(
        matriz_usuario_producto.columns[matriz_usuario_producto.iloc[idx_usuario] > 0]
    )
    puntajes = {}

    for idx in usuarios_similares:
        usuario_sim = usuarios_ids[idx]
        productos_sim = matriz_usuario_producto.columns[
            matriz_usuario_producto.iloc[idx] > 0
        ]
        for p in productos_sim:
            if p not in productos_usuario:
                puntajes[p] = puntajes.get(p, 0) + similitudes[idx]

    if not puntajes:
        return {"mensaje": "Sin recomendaciones personalizadas."}

    top = sorted(puntajes.items(), key=lambda x: x[1], reverse=True)[:cantidad]
    recomendados = [int(p[0]) for p in top]
    recomendados_info = info_productos(recomendados)

    return {
        "modelo": "user-based",
        "usuario": usuario_id,
        "recomendados": recomendados_info
    }


# =========================
# HÍBRIDO
# =========================
@app.get("/recomendaciones/hibrido/{usuario_id}/{producto_id}")
def recomendar_hibrido(usuario_id: int, producto_id: int, cantidad: int = 5, alpha: float = 0.5):
    """
    alpha controla el peso de cada modelo:
    - 0.0 = solo user-based
    - 1.0 = solo item-based
    """
    # item-based
    rec_item = recomendar_item_based(producto_id, cantidad * 2)
    items_item = [p["id"] for p in rec_item.get("recomendados", [])]

    # user-based
    rec_user = recomendar_user_based(usuario_id, cantidad * 2)
    items_user = [p["id"] for p in rec_user.get("recomendados", [])]

    puntaje = {}
    for i, pid in enumerate(items_item):
        puntaje[pid] = puntaje.get(pid, 0) + alpha * (len(items_item) - i)
    for i, pid in enumerate(items_user):
        puntaje[pid] = puntaje.get(pid, 0) + (1 - alpha) * (len(items_user) - i)

    top = sorted(puntaje.items(), key=lambda x: x[1], reverse=True)[:cantidad]
    recomendados = [int(p[0]) for p in top]
    recomendados_info = info_productos(recomendados)

    base = info_productos([producto_id])[0] if producto_id in productos_info["id"].values else {"id": producto_id}

    return {
        "modelo": "hibrido",
        "usuario": usuario_id,
        "producto_base": base,
        "recomendados": recomendados_info,
        "alpha": alpha
    }


#  Reentrenar manualmente
@app.get("/recomendaciones/reentrenar")
def reentrenar():
    entrenar_modelos()
    return {"message": " Modelos reentrenados con éxito"}



# =========================
# PREDICCIÓN DE STOCK
# =========================
@app.get("/stock-predictions")
def obtener_predicciones_stock():
    """
    Predice cuándo se agotará el stock de cada producto
    basándose en el histórico de ventas de los últimos 30 días.
    """
    from datetime import datetime, timedelta
    
    print("📊 Analizando predicciones de stock...")
    
    # Fecha de hace 30 días
    fecha_inicio = datetime.now() - timedelta(days=30)
    
    # Query para obtener ventas de los últimos 30 días con información del producto
    query = f"""
    SELECT 
        p.id AS producto_id,
        p.name AS producto_nombre,
        p.stock AS stock_actual,
        p."stock_minimo" AS stock_minimo,
        p.price AS precio,
        p."urlImage" AS imagen,
        COUNT(op.id) AS total_ordenes,
        SUM(op.quantity) AS total_vendido
    FROM product AS p
    LEFT JOIN order_product AS op ON p.id = op."productId"
    LEFT JOIN "order" AS o ON op."orderId" = o.id
    WHERE o."createdAt" >= '{fecha_inicio.strftime("%Y-%m-%d")}'
    GROUP BY p.id, p.name, p.stock, p."stock_minimo", p.price, p."urlImage"
    HAVING p.stock IS NOT NULL AND p.stock > 0
    ORDER BY p.stock ASC;
    """
    
    df = pd.read_sql(query, engine)
    
    if df.empty:
        return {
            "message": "No hay suficientes datos para generar predicciones",
            "predictions": []
        }
    
    predicciones = []
    
    for _, row in df.iterrows():
        producto_id = int(row['producto_id'])
        producto_nombre = row['producto_nombre']
        stock_actual = float(row['stock_actual']) if row['stock_actual'] else 0
        stock_minimo = float(row['stock_minimo']) if row['stock_minimo'] else 15
        total_vendido = float(row['total_vendido']) if row['total_vendido'] and pd.notna(row['total_vendido']) else 0
        
        # Calcular ventas diarias promedio
        ventas_diarias_promedio = total_vendido / 30 if total_vendido > 0 else 0
        
        # Predecir días hasta agotamiento
        if ventas_diarias_promedio > 0:
            dias_hasta_agotamiento = stock_actual / ventas_diarias_promedio
        else:
            dias_hasta_agotamiento = 999  # Sin ventas recientes
        
        # Determinar nivel de alerta
        if dias_hasta_agotamiento <= 3 or stock_actual <= stock_minimo:
            alert_level = "critical"
        elif dias_hasta_agotamiento <= 7:
            alert_level = "warning"
        else:
            alert_level = "normal"
        
        # Calcular cantidad sugerida de compra (para 30 días)
        cantidad_sugerida = max(int(ventas_diarias_promedio * 30), int(stock_minimo))
        
        # Solo incluir productos con alertas o con ventas recientes
        if alert_level in ["critical", "warning"] or total_vendido > 0:
            predicciones.append({
                "productId": producto_id,
                "productName": producto_nombre,
                "currentStock": stock_actual,
                "stockMinimo": stock_minimo,
                "dailySalesAverage": round(ventas_diarias_promedio, 2),
                "totalSoldLast30Days": total_vendido,
                "daysUntilStockout": round(dias_hasta_agotamiento, 1),
                "alertLevel": alert_level,
                "suggestedPurchaseQuantity": cantidad_sugerida,
                "price": float(row['precio']) if row['precio'] else 0,
                "image": row['imagen'] if row['imagen'] else ""
            })
    
    # Ordenar por criticidad (crítico primero)
    orden_alerta = {"critical": 0, "warning": 1, "normal": 2}
    predicciones.sort(key=lambda x: (orden_alerta[x["alertLevel"]], x["daysUntilStockout"]))
    
    print(f"✅ Generadas {len(predicciones)} predicciones de stock")
    
    return {
        "message": "Predicciones generadas exitosamente",
        "totalProducts": len(predicciones),
        "criticalAlerts": len([p for p in predicciones if p["alertLevel"] == "critical"]),
        "warningAlerts": len([p for p in predicciones if p["alertLevel"] == "warning"]),
        "predictions": predicciones
    }


# ==========================================
# 🚀 Punto de entrada para Railway
# ==========================================
if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
