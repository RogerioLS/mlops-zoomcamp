
GREEN = "\033[32;1m"
RED = "\033[31;1m"
CYAN = "\033[36;1m"
WHITE = "\033[37;1;4m"
COLOR_LIMITER = "\033[0m"

all:
# Alvo para rodar o contêiner
run:
	docker-compose up --build
	@echo $(CYAN)" --------------------------------------- "$(COLOR_LIMITER)
	@echo $(CYAN)"|  Docker run completed successfully!!  |"$(COLOR_LIMITER)
	@echo $(CYAN)" --------------------------------------- "$(COLOR_LIMITER)

# Alvo para parar e remover o contêiner
clean:
	docker-compose down --rmi all