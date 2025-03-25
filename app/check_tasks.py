import asyncio
import psutil
import os
from datetime import datetime
from collections import defaultdict


class TaskMonitor:
    def __init__(self, update_interval=10):
        self.update_interval = update_interval
        self.task_stats = defaultdict(dict)
        self.process = psutil.Process(os.getpid())

    async def start(self):
        """Запуск мониторинга в фоновом режиме"""
        asyncio.create_task(self._monitor_loop(), name="TaskMonitor")

    async def _monitor_loop(self):
        """Основной цикл сбора статистики"""
        while True:
            try:
                await self._collect_stats()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                print(f"Ошибка в мониторе: {e}")
                await asyncio.sleep(5)  # Пауза перед повторной попыткой

    async def _collect_stats(self):
        """Сбор статистики по всем задачам"""
        current_tasks = asyncio.all_tasks()
        tasks = [t for t in current_tasks if t is not asyncio.current_task()]

        # Сначала собираем CPU время до проверки задач
        cpu_times_before = {}
        for task in tasks:
            try:
                cpu_times_before[task] = self.process.cpu_times()
            except:
                pass

        # Даем время для накопления статистики
        await asyncio.sleep(0.1)

        # Собираем CPU время после
        cpu_times_after = {}
        for task in tasks:
            try:
                cpu_times_after[task] = self.process.cpu_times()
            except:
                pass

        # Обновляем статистику по задачам
        for task in tasks:
            try:
                task_name = task.get_name() or str(task.get_coro()).split(' ')[0]
                coro_info = str(task.get_coro())

                # Вычисляем использование CPU
                cpu_usage = 0.0
                if task in cpu_times_before and task in cpu_times_after:
                    before = cpu_times_before[task]
                    after = cpu_times_after[task]
                    cpu_usage = (after.user - before.user + after.system - before.system) / 0.1 * 100

                # Проверяем состояние задачи
                if task.done():
                    status = 'done'
                    try:
                        exception = str(task.exception()) if task.exception() else None
                    except asyncio.InvalidStateError:
                        exception = "No exception info"
                else:
                    status = 'running'
                    exception = None

                # Обновляем статистику задачи
                self.task_stats[task_name] = {
                    'coroutine': coro_info[:120] + '...' if len(coro_info) > 120 else coro_info,
                    'status': status,
                    'cpu_usage': cpu_usage,
                    'last_seen': datetime.now().isoformat(),
                    'exception': exception
                }
            except Exception as e:
                print(f"Ошибка при сборе статистики для задачи: {e}")

    def get_stats(self):
        """Возвращает текущую статистику"""
        # Общая информация о системе
        system_info = {
            'total_tasks': len(self.task_stats),
            'system_cpu': psutil.cpu_percent(),
            'process_cpu': self.process.cpu_percent(),
            'process_memory': self.process.memory_info().rss / 1024 / 1024,  # в MB
            'active_tasks': sum(1 for stat in self.task_stats.values() if stat.get('status') == 'running')
        }

        return {
            'system': system_info,
            'tasks': dict(self.task_stats)
        }

    def print_stats(self):
        """Выводит статистику в консоль"""
        try:
            stats = self.get_stats()
            sys_info = stats['system']

            print(f"\n{'=' * 50}")
            print(f"Мониторинг задач - {datetime.now().strftime('%H:%M:%S')}")
            print(f"Всего задач: {sys_info['total_tasks']} (активных: {sys_info['active_tasks']})")
            print(f"CPU: процесс {sys_info['process_cpu']:.1f}%, система {sys_info['system_cpu']:.1f}%")
            print(f"Память: {sys_info['process_memory']:.2f} MB")
            print(f"{'=' * 50}")

            for task_name, task_info in stats['tasks'].items():
                print(f"\nЗадача: {task_name}")
                print(f"Статус: {task_info.get('status', 'unknown')}")
                print(f"CPU: {task_info.get('cpu_usage', 0):.2f}%")
                print(f"Корутина: {task_info.get('coroutine', 'unknown')}")
                if task_info.get('exception'):
                    print(f"Ошибка: {task_info['exception']}")
                print(f"Последняя активность: {task_info.get('last_seen', 'unknown')}")
        except Exception as e:
            print(f"Ошибка при выводе статистики: {e}")