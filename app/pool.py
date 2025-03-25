import asyncio
from collections import deque


class SuspendedTaskPool:
    def __init__(self, max_concurrent=5):
        self.max_concurrent = max_concurrent
        self.tasks = deque()
        self.active_tasks = set()

    async def add_task(self, agen_func, *args):
        """Добавляем задачу-генератор в пул"""
        agen = agen_func(*args)
        # Получаем первую точку приостановки
        initial_awaitable = await agen.asend(None)
        self.tasks.append((agen, initial_awaitable))

    async def run(self):
        """Запускаем пул с циклическим выполнением задач"""
        while True:
            # Запускаем задачи, пока не достигнем лимита
            while self.tasks and len(self.active_tasks) < self.max_concurrent:
                agen, awaitable = self.tasks.popleft()
                task = asyncio.create_task(self._run_task(agen, awaitable))
                self.active_tasks.add(task)
                task.add_done_callback(self.active_tasks.discard)

            # Ждем завершения хотя бы одной задачи
            if self.active_tasks:
                await asyncio.wait(self.active_tasks,
                                   return_when=asyncio.FIRST_COMPLETED)

    async def _run_task(self, agen, awaitable):
        """Выполняет одну итерацию задачи и возвращает ее обратно в пул"""
        try:
            # Выполняем до следующей точки приостановки
            next_awaitable = await agen.asend(await asyncio.wait_for(awaitable, timeout=14400))
            self.tasks.append((agen, next_awaitable))
        except StopAsyncIteration:
            # Задача завершилась (не наш случай для бесконечных задач)
            pass
        except Exception as e:
            print(f"Ошибка в задаче: {e}")
            # Для бесконечных задач можно перезапустить
            self.tasks.append((agen, await agen.asend(None)))


# Пример бесконечной задачи с точками приостановки
async def sample_task(task_id):
    state = 0
    while True:
        print(f"Задача {task_id} работает, состояние {state}")

        # Точка приостановки (например, ожидание IO)
        await asyncio.sleep(0.5 + task_id * 0.2)

        # Изменяем состояние
        state += 1

        # Уступаем место другим задачам (точка приостановки)
        next_step = asyncio.sleep(0.1)  # Можно заменить на реальное ожидание
        received = yield next_step
        if received is not None:
            # Можно обработать внешние данные
            pass


