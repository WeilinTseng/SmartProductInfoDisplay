import sqlalchemy as sa
from sqlalchemy.orm import declarative_base, sessionmaker

def rData(class_name):
    BASE = declarative_base()

    class Product(BASE):

        __tablename__ = 'productInfor'

        id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)

        storeName = sa.Column(sa.String(64))
        productName = sa.Column(sa.String(64), unique = True)
        logo = sa.Column(sa.String(64), unique = True)
        price = sa.Column(sa.String(64))
        discount =sa.Column(sa.String(64))
        info = sa.Column(sa.String(64))
        web = sa.Column(sa.String(512))

        def __repr__(self):
            return f"id={self.id}, storeName={self.storeName}, productName={self.productName}, logo={self.logo}, price={self.price}, discount={self.discount}, info={self.info}, web={self.web}"
        


    engine = sa.create_engine("mysql+pymysql://root:@localhost:3306/productInfor") 
    # mysql+pymysql : 使用數據庫的driver
    # first root : username(不用動)
    # second root : password(密碼)
    # mysql+pymysql : 地址與端口
    # productInfor : 連接的數據庫

    session = sessionmaker(bind=engine)

    BASE.metadata.create_all(engine)

    session = session()

    # users = session.query(Product)
    # for u in users:
    #     print(u)

    object = class_name

    users = session.query(Product.storeName, Product.productName, Product.logo, Product.price, Product.discount, Product.info, Product.web).filter(Product.productName == object)

    for u in users:
        u = list(u)

        return u

