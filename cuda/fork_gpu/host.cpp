#include <sys/wait.h>
#include <unistd.h>
#include <string>
#include <vector>

#define KERNEL_PATH "/home/robert/tmp/kernel"
#define NORMAL "0"
#define BROKEN "1"

int main(int argc, char *argv[])
{
	int DEV_NUM = 2;
	int LOOP_NUM = 500;
	char *end;

	if (argc >= 2) {
		DEV_NUM = strtol(argv[1], &end, 10);
	}

	if (argc >= 3) {
		LOOP_NUM = strtol(argv[2], &end, 10);
	}

	std::vector<pid_t> child_pids;
	pid_t pid;
	for (int loop = 0; loop < LOOP_NUM; ++loop) {
		printf("==================== LOOP %d ====================\n", loop);

		while (child_pids.size() < DEV_NUM) {
			pid = fork();
			if (pid > 0) {
				printf("[%d]: Fork child%ld %d\n", getpid(), child_pids.size(), pid);
				child_pids.push_back(pid);
			} else if (pid < 0) {
				printf("[%d]: Failed to fork child%ld, rc = %d\n", getpid(), child_pids.size(), pid);
				return -1;
			} else {
				// Children break here
				break;
			}
		}

		if (child_pids.size() == DEV_NUM) {
			printf("[%d]: I'm parent, I have %ld children\n", getpid(), child_pids.size());
		} else {
			int ret;
			int child_idx = child_pids.size();
			if (loop % DEV_NUM == child_idx)
				exit(execl(KERNEL_PATH, std::to_string(child_idx).c_str(), BROKEN, NULL));
			else
				exit(execl(KERNEL_PATH, std::to_string(child_idx).c_str(), NORMAL, NULL));
		}

		printf("[%d]: Parent is waiting for it's children\n", getpid());

		int relaunch = 0;
		while (1) {
			int wstatus = 0;
			int exit_status = 0;
			int rc = wait(&wstatus);

			exit_status = WEXITSTATUS(wstatus);
			if (exit_status) {
				printf("[%d]: Child[%d] return error %d\n", getpid(), rc, exit_status);
				++relaunch;
			}

			if (rc > 0)
				continue;
			break;
		}

		child_pids.clear();
		if (!relaunch || relaunch >= DEV_NUM) {
			printf("[%d]: Unexpected error!!!!!!!\n", getpid());
			return -1;
		}
	}

	return 0;
}
