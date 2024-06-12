import sqlalchemy as sa
from sqlalchemy.orm import declarative_base, sessionmaker
import logging

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定義基礎類別，用於宣告模型
BASE = declarative_base()

# 定義產品類別，對應資料庫中的資料表
class Product(BASE):
    __tablename__ = 'productInfor'

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    storeName = sa.Column(sa.String(64))
    productName = sa.Column(sa.String(64), unique=True)
    logo = sa.Column(sa.String(64), unique=True)
    price = sa.Column(sa.String(64))
    discount = sa.Column(sa.String(64))
    info = sa.Column(sa.String(64))
    web = sa.Column(sa.String(512))

# 資料庫連接字串
DATABASE_URL = "mysql+pymysql://root:@localhost:3306/productInfor"
# mysql+pymysql : 使用數據庫的driver
# first root : username(不用動)
# second root : password(密碼)
# mysql+pymysql : 地址與端口
# productInfor : 連接的數據庫


# 創建資料庫引擎
engine = sa.create_engine(DATABASE_URL)

# 創建配置的Session類
Session = sessionmaker(bind=engine)

# 在資料庫中創建表
BASE.metadata.create_all(engine)

# 定義產品實例列表
products = [
    Product(storeName='https://i.postimg.cc/3RVVcwxb/Family-Mart.png', productName='Coca-Cola', logo='https://i.postimg.cc/pXYcDCzS/Coca-Cola.png', price='29', discount='Buy one get one free', info='https://i.postimg.cc/GtnzDT2r/info-Icon.png', web='https://www.coca-cola.com/tw/zh'),
    Product(storeName='https://i.postimg.cc/3RVVcwxb/Family-Mart.png', productName='Doritos', logo='https://i.postimg.cc/rwwj5k8d/Doritos.png', price='25', discount='Buy one get one 40% off', info='https://i.postimg.cc/GtnzDT2r/info-Icon.png', web='https://www.doritos.com/'),
    Product(storeName='https://i.postimg.cc/3RVVcwxb/Family-Mart.png', productName='Fanta', logo='https://i.postimg.cc/0ykWZSCy/Fanta.png', price='29', discount='Buy one get one free', info='https://i.postimg.cc/GtnzDT2r/info-Icon.png', web='https://www.coca-cola.com/tw/zh/brands/fanta'),
    Product(storeName='https://i.postimg.cc/3RVVcwxb/Family-Mart.png', productName='Good-Luck', logo='https://i.postimg.cc/ht2CYGCb/Good-Luck.png', price='20', discount='Buy one get one 40% off', info='https://i.postimg.cc/GtnzDT2r/info-Icon.png', web='https://www.kuai.com.tw/web/product/product.jsp?lang=tw&dm_id=DM1703056083466'),
    Product(storeName='https://i.postimg.cc/3RVVcwxb/Family-Mart.png', productName='Lays-original', logo='https://i.postimg.cc/s28TVNCM/Lay-s-original.png', price='35', discount='Buy one get one 40% off', info='https://i.postimg.cc/GtnzDT2r/info-Icon.png', web='https://www.lays.com/products/lays-classic-potato-chips'),
    Product(storeName='https://i.postimg.cc/3RVVcwxb/Family-Mart.png', productName='Liquor', logo='https://i.postimg.cc/PqN6nk9G/Liquor.png', price='65', discount='Regular price', info='https://i.postimg.cc/GtnzDT2r/info-Icon.png', web='https://www.strongzero.ch/product-page/strong-zero-grapes-case-24x500ml'),
    Product(storeName='https://i.postimg.cc/3RVVcwxb/Family-Mart.png', productName='Mountain-Dew', logo='https://i.postimg.cc/B6KMtBJM/Mountain-Dew.png', price='29', discount='Buy one get one free', info='https://i.postimg.cc/GtnzDT2r/info-Icon.png', web='https://www.pepsicopartners.com/pepsico/en/USD/PEPSICO-BRANDS/MTN-DEW%C2%AE/c/brand_mtnDew'),
    Product(storeName='https://i.postimg.cc/3RVVcwxb/Family-Mart.png', productName='Oreo', logo='https://i.postimg.cc/SRLdfQ29/Oreo.png', price='45', discount='Regular price', info='https://i.postimg.cc/GtnzDT2r/info-Icon.png', web='https://www.oreo.com/'),
    Product(storeName='https://i.postimg.cc/3RVVcwxb/Family-Mart.png', productName='Pocky', logo='https://i.postimg.cc/ZY6wSH7q/Pocky.png', price='45', discount='Regular price', info='https://i.postimg.cc/GtnzDT2r/info-Icon.png', web='https://www.glicoshop.com/?gad_source=1&gclid=CjwKCAjwgpCzBhBhEiwAOSQWQdHgmPWL8OQut-UOvRR7N0xWrEdbSslpnnzO-zmRtYMmUM_fEwXTMxoC01cQAvD_BwE'),
    Product(storeName='https://i.postimg.cc/3RVVcwxb/Family-Mart.png', productName='Pringles', logo='https://i.postimg.cc/7YVXC6y8/Pringles.jpg', price='50', discount='Regular price', info='https://i.postimg.cc/GtnzDT2r/info-Icon.png', web='https://www.pringles.com/en-us/home.html'),
    Product(storeName='https://i.postimg.cc/3RVVcwxb/Family-Mart.png', productName='Sprite', logo='https://i.postimg.cc/cLkmmbHF/Sprite.png', price='29', discount='Buy one get one free', info='https://i.postimg.cc/GtnzDT2r/info-Icon.png', web='https://www.coca-cola.com/tw/zh/brands/sprite'),
    Product(storeName='https://i.postimg.cc/3RVVcwxb/Family-Mart.png', productName='Water', logo='https://i.postimg.cc/FsmDsS8V/Water.png', price='20', discount='Regular price', info='https://i.postimg.cc/GtnzDT2r/info-Icon.png', web='https://zh.wikipedia.org/wiki/%E6%B0%B4'),
]

# 將產品添加到資料庫
try:
    session = Session()
    session.add_all(products)
    session.commit()
    logger.info("Products added successfully!")
except Exception as e:
    logger.error(f"Error adding products: {e}")
    session.rollback()
finally:
    session.close()