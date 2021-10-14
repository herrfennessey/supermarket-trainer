from albertsons_category_mapping import albertsons_mapping


class Labeler:
    def __init__(self, category_mapping):
        self.mapping = category_mapping

    def label_product(self, category_list):
        # Trash data
        if len(category_list) == 1:
            return None

        # Start from the deepest breadcrumb and move backwards
        category_list.reverse()

        for category in category_list:
            if category in self.mapping and self.mapping.get(category):
                return self.mapping.get(category)

        return None


class AlbertsonsLabeler(Labeler):
    def __init__(self):
        super().__init__(albertsons_mapping)
