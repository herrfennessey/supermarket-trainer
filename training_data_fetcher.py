from firestore_dao import FirestoreDao

from labeler import AlbertsonsLabeler


class TrainingDataFetcher:
    def __init__(self):
        self.firestore_db = FirestoreDao()
        self.labeler = AlbertsonsLabeler()

    def fetch(self):
        all_searches = self.firestore_db.get_all_searches()
        all_products = self.firestore_db.get_all_products()

        print("processing PIDs into a dictionary")
        pid_dict = {str(pid.id): pid.to_dict() for pid in all_products}

        searches_with_labels = []
        for search in all_searches:
            for idx, search_result in enumerate(search.get("search_results")):
                pid = search_result.get("document_reference").id
                if pid in pid_dict:
                    product = pid_dict.get(pid)
                    label = self.labeler.label_product(product.get("product_category"))
                    if label:
                        searches_with_labels.append({
                            "sentence": search.id,
                            "label": label,
                            "page_rank": idx
                        })
                else:
                    # Can't find it in the pid collection - just skip
                    continue
        return searches_with_labels
