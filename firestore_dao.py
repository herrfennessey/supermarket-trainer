import logging

from google.cloud import firestore

SEARCHES_COLLECTION = "searches-v1"
PRODUCTS_COLLECTION = "products-v1"

logger = logging.getLogger()


class FirestoreDao:
    def __init__(self):
        self.firestore_db = firestore.Client()

    def get_all_searches(self):
        print("Fetching all searches")
        searches = self.firestore_db.collection(SEARCHES_COLLECTION).get()
        print(f"Fetched {len(searches)} searches from DB")
        return searches

    def get_all_products(self):
        print("Fetching all products")
        products = self.firestore_db.collection(PRODUCTS_COLLECTION).get()
        print(f"Fetched {len(products)} products from DB")
        return products
