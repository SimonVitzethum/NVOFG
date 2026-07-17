set pagination off
set confirm off
set disable-randomization on
set follow-fork-mode child
set detach-on-fork off
handle SIGSEGV nostop noprint pass
handle SIGUSR1 nostop noprint pass
# stop just before the FG requirements call — _nvngx base is known by now
break s5_host.c:608
run
# anchor all breakpoints to the measured base (ASLR-proof)
set $b = (unsigned long long)g_dbg_nvngx_base
printf "=== _nvngx base = 0x%llx ===\n", $b
# return-value capture points inside the evaluator 0x18000c1e0 and its parent, in call order:
break *($b+0xc254)
commands
  silent
  printf ">> after call 0x180061730 (nvapi driver-version check): eax=0x%x\n", (unsigned)$eax
  continue
end
break *($b+0xc283)
commands
  silent
  printf ">> after call 0x18003b370 (string/json validator): eax=0x%x\n", (unsigned)$eax
  continue
end
break *($b+0xc40e)
commands
  silent
  printf ">> after call 0x18000b720: eax=0x%x\n", (unsigned)$eax
  continue
end
break *($b+0xe185)
commands
  silent
  printf "== EVALUATOR 0x18000c1e0 RETURNED: eax=0x%x ==\n", (unsigned)$eax
  continue
end
break *($b+0xe1d6)
commands
  silent
  printf ">> per-feature DISPATCH about to call: target=0x%llx  (rva-from-nvngx=0x%llx)\n", (unsigned long long)$rax, ((unsigned long long)$rax)-$b
  printf "   args: rcx(inst)=0x%llx rdx(pd)=0x%llx r8(fdi)=0x%llx r9(req)=0x%llx\n", (unsigned long long)$rcx,(unsigned long long)$rdx,(unsigned long long)$r8,(unsigned long long)$r9
  # step into the dispatched fn and show its first ~40 instructions
  stepi
  printf "   dispatched fn entry:\n"
  x/40i $pc
  continue
end
break *($b+0xe1dc)
commands
  silent
  printf ">> after per-feature dispatch [0x114838]: eax=0x%x\n", (unsigned)$eax
  continue
end
continue
quit
