import logging

from google.cloud import firestore

SEARCHES_COLLECTION = "searches-v1"
PRODUCTS_COLLECTION = "products-v1"

logger = logging.getLogger()


class FirestoreDao:
    def __init__(self):
        self.firestore_db = firestore.Client()

    def get_example_search(self, search_term="arizona"):
        doc_ref = self._get_search_reference(search_term)
        doc = doc_ref.get()
        return doc

    def get_products(self, pid_list: list[str]):
        ref_list = [self._get_product_reference(pid) for pid in pid_list]
        return self.firestore_db.get_all(ref_list)

    def _get_search_reference(self, search_term: str):
        return self.firestore_db.collection(SEARCHES_COLLECTION).document(search_term)

    def _get_product_reference(self, pid: str):
        return self.firestore_db.collection(PRODUCTS_COLLECTION).document(pid)

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
