class RolloutBuffer:
    def __init__(self):
        self.clear()

    def add(self, *transition):
        self.data.append(transition)

    def clear(self):
        self.data = []
