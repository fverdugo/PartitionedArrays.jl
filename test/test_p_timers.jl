
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
  print_timer(t,"timer.txt","w")
  print_timer(t,"timer.txt","a")
  print_timer(t,"timer.txt";write=true,create=false)
  print_timer(t,"timer.txt";append=true)

  print_csv(parts,2.0,"value","results.txt",write=true)
  print_csv(parts,3.0,"another value","results.txt",append=true)
  print_csv(parts,4.0,"still another value","results.txt","a")

end

