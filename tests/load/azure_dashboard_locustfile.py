from locust import HttpUser, between, task


class AzureDashboardUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def open_dashboard(self):
        with self.client.get(
            "/",
            name="Dashboard homepage",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(
                    f"Unexpected status code: {response.status_code}"
                )
