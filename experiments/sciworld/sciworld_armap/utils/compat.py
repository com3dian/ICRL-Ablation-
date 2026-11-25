"""ScienceWorld API and JVM lifecycle compatibility shared by both runners."""

LEGACY_METHODS = {
    "get_task_names": "getTaskNames",
    "get_num_moves": "getNumMoves",
    "get_valid_action_object_combinations": "getValidActionObjectCombinations",
    "get_possible_objects": "getPossibleObjects",
}


def install_scienceworld_compat(env_class):
    if getattr(env_class, "_icrl_compat_installed", False):
        return
    for legacy, current in LEGACY_METHODS.items():
        if not hasattr(env_class, legacy):
            if not hasattr(env_class, current):
                raise RuntimeError(f"ScienceWorld has neither {legacy} nor {current}")
            setattr(env_class, legacy, getattr(env_class, current))
    original_init = env_class.__init__
    original_close = getattr(env_class, "close", None)
    env_class._icrl_instances = set()

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        env_class._icrl_instances.add(self)

    def close(self):
        if not getattr(self, "_icrl_closed", False):
            if original_close:
                original_close(self)
            else:
                self._gateway.shutdown()
            self._icrl_closed = True
            env_class._icrl_instances.discard(self)

    env_class.__init__ = initialize
    env_class.close = close
    env_class._icrl_compat_installed = True
