library(dplyr)
library(tidyr)

# MCS for tourism dataset
Methods <- c("QOpt($\\beta=100$)", "base", "ols", "wls", "shr", "sam")
Methods_name <- c("QOpt", "Base", "OLS", "MinT(var)", "MinT(shr)", "MinT(sam)")
Scenarios <- c("normal", "skew")
ALPHAs <- c(0.05, 0.2, 0.8, 0.95)
output <- list()
for (scenario in Scenarios) {
  table_path <- paste0("output/logs/tourism_", scenario, ".csv")
  tb <- read.csv(table_path)

  tbl_average <- tb |>
    group_by(alpha, method, window) |>
    summarise(loss = mean(loss))

  output[[scenario]] <- list()
  # mcs

  for (alpha in sort(unique(tbl_average$alpha))) {
    mcs_results <- tbl_average |>
      filter(alpha == .env$alpha) |>
      pivot_wider(names_from = "method", values_from = "loss") |>
      ungroup() |>
      select(-alpha, -window) |>
      as.matrix() |>
      MCS::MCSprocedure(alpha = 0.05)
    included <- mcs_results@Info$included
    col_results <- tbl_average |>
      filter(alpha == .env$alpha) |>
      group_by(method) |>
      summarise(loss = mean(loss)) |>
      arrange(factor(method, levels = Methods)) |>
      pull(loss)
    output[[scenario]][[paste0(alpha)]] <- list(included = included, loss = col_results)
  }
}


for (scenario in Scenarios) {
  s_list <- output[[scenario]]
  total_text <- ""
  for (m_idx in seq_along(Methods)) {
    m <- Methods[m_idx]
    m_name <- Methods_name[m_idx]
    line_text <- paste0(m_name)
    for (alpha in ALPHAs) {
      val <- s_list[[paste0(alpha)]]$loss[m_idx]
      text <- sprintf("%.3f", val)
      if (min(s_list[[paste0(alpha)]]$loss) == val) {
        text <- sprintf("\\textbf{%s}", text)
      }
      if (m %in% s_list[[paste0(alpha)]]$included) {
        if (length(s_list[[paste0(alpha)]]$included) < length(Methods)) {
          text <- sprintf("%s\\textsuperscript{*}", text)
        }
      }
      line_text <- paste0(line_text, " & ", text)
    }
    if (m_idx < length(Methods)) {
      line_text <- paste0(line_text, "\\\\\n")
    }
    total_text <- paste0(total_text, line_text)
  }
  cat(total_text, file = sprintf("output/tables/tourism_%s.tex", scenario))
}


output <- list()
for (scenario in Scenarios) {
  table_path <- paste0("output/logs/tourism_", scenario, "_insample.csv")
  tb <- read.csv(table_path)

  tbl_average <- tb |>
    group_by(alpha, method, window) |>
    summarise(loss = mean(loss))

  output[[scenario]] <- list()
  # mcs

  for (alpha in sort(unique(tbl_average$alpha))) {
    mcs_results <- tbl_average |>
      filter(alpha == .env$alpha) |>
      pivot_wider(names_from = "method", values_from = "loss") |>
      ungroup() |>
      select(-alpha, -window) |>
      as.matrix() |>
      MCS::MCSprocedure(alpha = 0.05)
    included <- mcs_results@Info$included
    col_results <- tbl_average |>
      filter(alpha == .env$alpha) |>
      group_by(method) |>
      summarise(loss = mean(loss)) |>
      arrange(factor(method, levels = Methods)) |>
      pull(loss)
    output[[scenario]][[paste0(alpha)]] <- list(included = included, loss = col_results)
  }
}


for (scenario in Scenarios) {
  s_list <- output[[scenario]]
  total_text <- ""
  for (m_idx in seq_along(Methods)) {
    m <- Methods[m_idx]
    m_name <- Methods_name[m_idx]
    line_text <- paste0(m_name)
    for (alpha in ALPHAs) {
      val <- s_list[[paste0(alpha)]]$loss[m_idx]
      text <- sprintf("%.3f", val)
      if (min(s_list[[paste0(alpha)]]$loss) == val) {
        text <- sprintf("\\textbf{%s}", text)
      }
      if (m %in% s_list[[paste0(alpha)]]$included) {
        if (length(s_list[[paste0(alpha)]]$included) < length(Methods)) {
          text <- sprintf("%s\\textsuperscript{*}", text)
        }
      }
      line_text <- paste0(line_text, " & ", text)
    }
    if (m_idx < length(Methods)) {
      line_text <- paste0(line_text, "\\\\\n")
    }
    total_text <- paste0(total_text, line_text)
  }
  cat(total_text, file = sprintf("output/tables/tourism_%s_insample.tex", scenario))
}
