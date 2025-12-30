.PHONY: up down logs test-backend test-e2e

COMPOSE ?= docker compose

up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=200

test-backend:
	$(COMPOSE) up -d postgres redis
	$(COMPOSE) run --rm backend pytest -q

test-e2e:
	$(COMPOSE) up -d --build
	cd frontend && npm ci
	cd frontend && npx playwright install --with-deps
	cd frontend && npm run test:e2e


