from dedalus.utils.parallelism import com_sys
class countcalls(object):
    def __init__(self):
        """
        A simple class that builds a dictionary of functions and
        their call counts

        """

        self.counters = {}

    def __call__(self, func):
        """Decorator to time function execution."""

        def wrapper(*args, **kwargs):
            retval = func(*args, **kwargs)
            try:
                self.counters[func.func_name] += 1
            except KeyError:
                self.counters[func.func_name] = 1

            return retval
        return wrapper

    def print_stats(self, proc=0):
        """Print cumulative times for functions executed on a specified processor."""

        if com_sys.myproc == proc:
            print
            print "---Call counts (proc %i)---" % (proc)
            for k,v in self.counters.iteritems():
                print "%s: %i calls" % (k,v)
            print

counts = countcalls()

if __name__ == "__main__":
    timer = Timer()

    @timer
    def sleep_two_sec():
        time.sleep(2.0)

    sleep_two_sec()
    timer.print_stats()
