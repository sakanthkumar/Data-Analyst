class AgentState:

    def __init__(self):

        self.dataset = None

        self.schema = {}

        self.analytics = {}

        self.insights = []

        self.recommendations = []

        self.report = None

        self.metadata = {}

        self.workflow_status = {
            "schema": False,
            "analytics": False,
            "insight": False,
            "report": False
        }