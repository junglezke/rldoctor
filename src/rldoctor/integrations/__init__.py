"""Framework integrations.

Each submodule imports its framework lazily, so ``rldoctor`` never drags a
training stack into an environment that only wants to read a log file.
"""
