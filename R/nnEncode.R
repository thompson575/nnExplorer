nnEncode <- function(f, fList, fNumber, label) {
  f     <- tolower(f)
  fList <- tolower(fList)
  code  <- rep(0, length(f))
  for( j in 1:length(f)) {
    len <- nchar(f[j])
    perfect <- nmatch <- match <- 0
    for(k in 1:length(fList)) {
      if( f[j] == substr(fList[k], 1, len) ) {
        if( len == nchar(fList[k])) perfect <- k
        match <- k
        nmatch <- nmatch + 1
      }
    }
    if( perfect != 0 ) code[j] = fNumber[perfect]
    else if( nmatch == 1 ) code[j] = fNumber[match]
    else if( nmatch == 0 ) stop(paste(label, f[j], "not recognised"))
    else stop(paste(label, f[j], "abbreviation not unique"))
  }
  return(code)
}