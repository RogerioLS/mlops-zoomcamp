GREEN = "\033[32;1m"
RED = "\033[31;1m"
CYAN = "\033[36;1;3;208m"
WHITE = "\033[37;1;4m"
COLOR_LIMITER = "\033[0m"

all: homework

homework:
	@echo $(CYAN)" ------------------ "$(COLOR_LIMITER)
	@echo $(CYAN)"| Start of answers |"$(COLOR_LIMITER)
	@echo $(CYAN)" ------------------ "$(COLOR_LIMITER)
	@python script/homework.py
	@echo $(CYAN)" ------------------ "$(COLOR_LIMITER)
	@echo $(CYAN)"|  End of answers  |"$(COLOR_LIMITER)
	@echo $(CYAN)" ------------------ "$(COLOR_LIMITER)

.PHONY: homework
