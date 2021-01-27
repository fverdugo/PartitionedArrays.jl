
using PartitionedArrays

function test_p_timers(parts)

  t = PTimer(parts)
  
  tic!(t)
  toc!(t,"Phase 1")
  toc!(t,"Phase 3")
  sleep(0.2)
  toc!(t,"Phase 143")
  tic!(t)
  sleep(0.4)
  toc!(t,"Matrix Assembly")
  
  display(t)
  print_timer(t,format=:csv)
  print_timer("timer.txt",t,format=:REPL)
  print_timer("timer.txt",t)

end

